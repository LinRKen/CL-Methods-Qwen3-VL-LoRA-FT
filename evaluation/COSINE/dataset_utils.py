"""Read existing COSINE inference output without importing the inference engine."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


TASKS = ("whole", "geo", "hypo", "motion", "nav", "pattern", "view")
TEST_DATASETS = tuple("cosine_{}_test".format(task) for task in TASKS)


def _reject_constant(value):
    raise ValueError("Non-finite JSON number: {}".format(value))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_constant)
            except ValueError as error:
                raise ValueError("{}:{}: {}".format(path, line_number, error)) from error
            if not isinstance(record, dict):
                raise TypeError("{}:{}: expected a JSON object".format(path, line_number))
            records.append(record)
    return records


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registered_test(dataset_name):
    if dataset_name not in TEST_DATASETS:
        raise ValueError("Choose one registered COSINE test split: {}".format(TEST_DATASETS))
    finetune_root = Path(__file__).resolve().parents[2] / "qwen-vl-finetune"
    sys.path.insert(0, str(finetune_root))
    # The registry is lightweight; do not import qwenvl.infer or media processors.
    from qwenvl.data import data_list

    config = data_list([dataset_name])[0]
    path = Path(config["annotation_path"]).expanduser().resolve(strict=True)
    if path.suffix.lower() == ".jsonl":
        records = read_jsonl(path)
    else:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle, parse_constant=_reject_constant)
    if not isinstance(records, list) or not records:
        raise ValueError("The annotation file must contain a non-empty record list")
    if any(not isinstance(record, dict) for record in records):
        raise TypeError("Every annotation must be an object")
    return records, path


def _conversations(sample):
    turns = sample.get("conversations")
    if not isinstance(turns, list) or any(not isinstance(turn, dict) for turn in turns):
        raise TypeError("Sample conversations must be a list of objects")
    return turns


def extract_ground_truth(sample):
    answers = [turn.get("value", turn.get("content")) for turn in _conversations(sample)
               if turn.get("from", turn.get("role")) in ("gpt", "assistant")]
    if len(answers) != 1 or not isinstance(answers[0], str) or not answers[0].strip():
        raise ValueError("Expected exactly one non-empty textual reference answer")
    return answers[0]


def extract_question(sample):
    questions = [turn.get("value", turn.get("content")) for turn in _conversations(sample)
                 if turn.get("from", turn.get("role")) in ("human", "user")]
    if not questions or any(not isinstance(question, str) for question in questions):
        raise ValueError("Expected textual human/user questions")
    question = "\n\n".join(questions)
    # Remove media markers only. Keep all options and all ordinary question text.
    for marker in ("<image>", "<video>", "<visual>", "<|vision_start|>",
                   "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"):
        question = question.replace(marker, "")
    if not question.strip():
        raise ValueError("Question is empty after removing media markers")
    return question.strip()


def align_predictions(predictions, samples, dataset_name):
    """Use the exact test-row index; repeated source IDs remain separate samples."""
    if not predictions:
        raise ValueError("The prediction file is empty")
    aligned = {}
    for prediction in predictions:
        index = prediction.get("inference_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(samples):
            raise ValueError("Invalid or missing inference_index: {!r}".format(index))
        if index in aligned:
            raise ValueError("Duplicate inference_index: {}".format(index))
        if prediction.get("dataset") != dataset_name:
            raise ValueError("Dataset mismatch at inference_index={}".format(index))
        sample = samples[index]
        meta = sample.get("_meta", {})
        if not isinstance(meta, dict) or prediction.get("_meta") != meta:
            raise ValueError("Annotation metadata mismatch at inference_index={}".format(index))
        expected = {
            "id": meta.get("id"),
            "sample_index": meta.get("sample_index", index),
            "task_type": meta.get("task_type"),
            "metric_type": meta.get("metric_type"),
            "ground_truth": extract_ground_truth(sample),
        }
        for name, value in expected.items():
            if name not in prediction or prediction[name] != value:
                raise ValueError("{} mismatch at inference_index={}".format(name, index))
        metric_type = expected["metric_type"]
        if metric_type not in ("MCQ", "OPEN", "MRA"):
            raise ValueError("Unsupported metric_type: {!r}".format(metric_type))
        if not isinstance(prediction.get("prediction"), str):
            raise TypeError("Prediction must be text at inference_index={}".format(index))
        aligned[index] = {
            **expected,
            "dataset": dataset_name,
            "inference_index": index,
            "source_dataset": meta.get("dataset"),
            "n_options": meta.get("n_options"),
            "_meta": meta,
            "question": extract_question(sample),
            # Preserve the complete generated answer, including any reasoning.
            "prediction": prediction["prediction"],
            "generation": prediction.get("generation"),
        }
    missing = sorted(set(range(len(samples))) - set(aligned))
    if missing:
        raise ValueError("Full-test evaluation requires every sample; missing {} indices, first: {}"
                         .format(len(missing), missing[:10]))
    return [aligned[index] for index in range(len(samples))]


def write_json_atomic(path, value, jsonl=False):
    """Replace a completed artifact; leave the previous artifact intact on failure."""
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as handle:
            temporary_path = Path(handle.name)
            if jsonl:
                for record in value:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
