# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.wandb_utils import (
    configure_wandb_gpu_monitoring,
    prepare_training_gpu_selection,
)

# Resolve GPU identities before torch/Transformers can initialize CUDA.
_training_gpu_selection = prepare_training_gpu_selection()

import torch
import transformers

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer, set_seed

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    configure_wandb_gpu_monitoring(training_args, _training_gpu_selection)
    resume = training_args.resume_checkpoint_path
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_absolute() or not (resume_path / "trainer_state.json").is_file():
            raise ValueError("resume_checkpoint_path must be an explicit absolute Trainer checkpoint")
        if model_args.adapter_name_or_path is not None:
            raise ValueError("Choose adapter-only continuation or full checkpoint restore, not both")
        if not training_args.lora_enable:
            raise ValueError("Explicit checkpoint restore requires LoRA")
    if model_args.adapter_name_or_path is not None and not training_args.lora_enable:
        raise ValueError("adapter_name_or_path requires LoRA")
    if training_args.lora_enable and model_args.tune_mm_vision:
        raise ValueError("LoRA training requires a frozen visual encoder")
    output = Path(training_args.output_dir)
    if any(output.glob("checkpoint-*")) or (output / "trainer_state.json").exists():
        raise ValueError("Refusing to reuse a training output; restore into a fresh output directory")
    final_output = Path(training_args.final_adapter_dir or training_args.output_dir)
    if training_args.final_adapter_dir and final_output.exists() and any(final_output.iterdir()):
        raise ValueError("Final adapter directory must be empty or absent")
    os.makedirs(training_args.output_dir, exist_ok=True)
    set_seed(training_args.seed)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor_path = model_args.processor_name_or_path or model_args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        processor_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, TaskType

        for parameter in model.parameters():
            parameter.requires_grad = False

        adapter_path = resume or model_args.adapter_name_or_path
        if adapter_path:
            adapter_dir = Path(adapter_path)
            if not adapter_dir.is_absolute() or not (adapter_dir / "adapter_config.json").is_file():
                raise ValueError("Adapter input must be an explicit absolute adapter directory")
            lora_config = PeftConfig.from_pretrained(adapter_path)
            expected = {"r": training_args.lora_r, "lora_alpha": training_args.lora_alpha,
                        "lora_dropout": training_args.lora_dropout}
            if (str(lora_config.peft_type) not in ("LORA", "PeftType.LORA")
                    or any(getattr(lora_config, key, None) != value for key, value in expected.items())
                    or getattr(lora_config, "bias", "none") != "none"
                    or getattr(lora_config, "modules_to_save", None)):
                raise ValueError("Inherited adapter is incompatible with the requested LoRA configuration")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            policy = "restore_full_checkpoint" if resume else "adapter_weights_with_fresh_training_state"
        else:
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                exclude_modules=r".*visual[.].*",
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            policy = "base_model_with_fresh_training_state"

        trainable = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        if not trainable or any("visual" in name.split(".") or "lora_" not in name
                                for name, _ in trainable):
            raise ValueError("Only language-model LoRA parameters may be trainable")
        logging.info("LoRA training policy: %s", policy)
        model.print_trainable_parameters()
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    # None starts a new optimizer/scheduler/epoch/step even when adapter weights were loaded.
    result = trainer.train(resume_from_checkpoint=resume)
    # Keep a complete final checkpoint even when the configured interval does not
    # land on the last step. All ranks participate in DeepSpeed checkpoint saving.
    trainer.accelerator.wait_for_everyone()
    final_checkpoint = output / f"checkpoint-{trainer.state.global_step}"
    needs_final_checkpoint = [not final_checkpoint.exists() if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(needs_final_checkpoint, src=0)
    if training_args.final_adapter_dir and needs_final_checkpoint[0]:
        trainer._save_checkpoint(trainer.model, trial=None)
    trainer.accelerator.wait_for_everyone()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=str(final_output))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(final_output))
        summary = {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "train_samples": len(data_module["train_dataset"]),
            "world_size": training_args.world_size,
            "trainable_parameters": sum(getattr(parameter, "ds_numel", parameter.numel())
                                        for parameter in model.parameters()
                                        if parameter.requires_grad),
            "training_state_policy": policy if training_args.lora_enable else "full_finetuning",
            "resume_checkpoint_path": resume,
            "adapter_name_or_path": model_args.adapter_name_or_path,
            "final_adapter_dir": str(final_output),
            "metrics": result.metrics,
        }
        with (output / "training_result.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
