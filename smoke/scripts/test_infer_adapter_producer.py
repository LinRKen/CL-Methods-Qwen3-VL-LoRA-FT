"""CPU regression for Adapter producer validation across infer settings.

Run from the repository root with the Qwen3VL-vLLM Python environment.
Fixtures contain dummy bytes, never load a model and never use CUDA.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from lib.artifacts import make_manifest
from lib.common import write_json_atomic
from lib.config import load_config
from lib.infer_executor import _inputs
from lib.records import (
    bind_inputs, create_experiment, create_job, create_stage_run, load_job,
    register_artifact, update_status,
)


class InferAdapterProducerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="infer-producer-cpu-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        base = root / "base"
        base.mkdir()
        write_json_atomic(base / "config.json", {})
        dataset = root / "dataset"
        (dataset / "motion").mkdir(parents=True)
        write_json_atomic(dataset / "motion" / "test.json", [])
        self.snapshot = load_config(REPO / "scripts/experiments/configs/multitask.yaml",
            cli_overrides={"experiment": {
                "records_root": str(root / "records"), "base_model": str(base),
                "processor_path": str(base), "dataset_root": str(dataset),
                "checkpoints_root": str(root / "checkpoints"),
                "infer_root": str(root / "infer"), "eval_root": str(root / "eval"),
                "logs_root": str(root / "logs"),
            }})
        self.experiment_dir = create_experiment(self.snapshot, experiment_id="producer-fixture")
        tune = create_stage_run(self.experiment_dir, "tune", run_id="tune-fixture")
        lineage = {"train_task": "whole", "initialization": "base_model"}
        self.source_dir = create_job(tune, "tune-whole", task="whole", lineage=lineage,
                                    output={"artifact_kind": "adapter", "slot": "whole"})
        self.base_manifest = make_manifest("base_model", base,
            source="configuration:experiment.base_model")
        source = bind_inputs(self.source_dir, {"base-model": self.base_manifest})
        update_status(self.source_dir, "running")
        adapter = root / "adapter"
        adapter.mkdir()
        params = self.snapshot["config"]["tune"]["params"]
        write_json_atomic(adapter / "adapter_config.json", {
            "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "r": params["lora_r"], "lora_alpha": params["lora_alpha"],
            "lora_dropout": params["lora_dropout"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "exclude_modules": r".*visual[.].*", "base_model_name_or_path": str(base),
        })
        (adapter / "adapter_model.safetensors").write_bytes(b"CPU fixture; not model weights")
        producer = {key: source[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
        self.adapter = make_manifest("adapter", adapter, producer=producer, metadata={
            "slot": "whole", "base_model_fingerprint": self.base_manifest["fingerprint"],
            "config_fingerprint": source["config_fingerprint"], "lineage": lineage,
            "training_state_policy": source["training_state_policy"],
        })
        register_artifact(self.source_dir, "adapter", self.adapter)
        update_status(self.source_dir, "succeeded")
        update_status(tune, "running")
        update_status(tune, "succeeded")
        infer = create_stage_run(self.experiment_dir, "infer", run_id="infer-fixture")
        specs = [
            {"role": "base-model", "kind": "base_model", "path": str(base),
             "source": "configuration:experiment.base_model", "metadata": {}},
            {"role": "processor", "kind": "processor", "path": str(base),
             "source": "configuration:experiment.processor_path", "metadata": {}},
            {"role": "annotation-motion-test", "kind": "annotation",
             "path": str(dataset / "motion/test.json"),
             "source": "configuration:experiment.dataset_root",
             "metadata": {"task": "motion", "split": "test"}},
        ]
        self.job_dir = create_job(infer, "infer-whole", tasks=["motion"], input_specs=specs,
            requires=[{"stage": "tune", "job_id": "tune-whole",
                       "artifact_kind": "adapter", "slot": "whole"}],
            lineage={"adapter_ref": "tune-whole", "test_tasks": ["motion"]},
            output={"artifact_kind": "prediction", "slot": "whole", "tasks": ["motion"]})
        self.job, self.job_snapshot = load_job(self.job_dir)

    def validate(self, setting):
        snapshot = deepcopy(self.job_snapshot)
        snapshot["config"]["experiment"]["setting"] = setting
        return _inputs(self.job_dir, self.job, snapshot, {"adapter": self.adapter},
                       self.experiment_dir)

    def test_multitask(self):
        job, inputs = self.validate("multitask")
        self.assertEqual(inputs["adapter"], self.adapter)
        self.assertEqual(load_job(self.job_dir)[0]["inputs"], inputs)
        self.assertEqual(job["status"], "planned")

    def test_singletask(self):
        self.validate("singletask")

    def test_continual_learning(self):
        self.validate("cl")

    def test_transfer_round1(self):
        self.validate("transfer_round1")

    def test_transfer_round2_local_source(self):
        self.validate("transfer_round2")

    def test_transfer_baseline_without_prediction_reuse(self):
        self.validate("transfer_baseline")

    def test_zero_shot_without_adapter(self):
        snapshot = deepcopy(self.job_snapshot)
        snapshot["config"]["experiment"]["setting"] = "zero_shot"
        job = deepcopy(self.job)
        job["requires"] = []
        job["lineage"] = {"model": snapshot["config"]["experiment"]["base_model"],
                          "adapter": None}
        # No real record mutation: this case checks the base-only validation branch.
        from unittest.mock import patch
        with patch("lib.infer_executor.bind_inputs", side_effect=lambda directory, inputs: job):
            _, inputs = _inputs(self.job_dir, job, snapshot, {}, self.experiment_dir)
        self.assertFalse(any(item["kind"] == "adapter" for item in inputs.values()))

    def test_unregistered_adapter_is_rejected(self):
        record_path = self.source_dir / "job.json"
        record = json.loads(record_path.read_text())
        record["artifacts"] = []
        write_json_atomic(record_path, record)
        with self.assertRaisesRegex(ValueError, "not registered by its claimed producer"):
            self.validate("multitask")
        self.assertEqual(load_job(self.job_dir)[0]["inputs"], {})

    def test_ineligible_producer_is_rejected(self):
        record_path = self.source_dir / "job.json"
        record = json.loads(record_path.read_text())
        record["status"] = "failed"
        write_json_atomic(record_path, record)
        with self.assertRaisesRegex(ValueError, "eligible registered producer"):
            self.validate("multitask")
        self.assertEqual(load_job(self.job_dir)[0]["inputs"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
