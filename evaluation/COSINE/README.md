# COSINE Qwen 柔性评测

读取已有 COSINE infer JSONL 和注册测试集 GT，用本地 Qwen3-VL 作纯文本 judge。该脚本是独立的单次评测入口；批量实验评测请使用 `scripts/experiments/run_eval.py`，其记录、资源调度和汇总规则见[实验编排说明](../../scripts/experiments/README.md)。

## 文件与输入

- `eval_cosine.py`：参数、GPU 身份检查、vLLM 批量判分、断点恢复和产物保存。
- `dataset_utils.py`：加载注册测试集、预测与标注对齐、原子写入。
- `qwen_judge.py`：MCQ/OPEN/MRA 提示词、JSON 协议和聚合计分。
- `eval_cosine.sh`：启动脚本。

支持 `cosine_whole_test` 及 geo、hypo、motion、nav、pattern、view 六个 test split。
输入来自 `qwenvl/infer/infer_cosine.py`，按 `inference_index` 对齐测试集行号，并核对 dataset、完整 `_meta`、GT 和样本字段。要求覆盖完整测试集；缺失/重复行、错集或标注不符会报错。重复 source ID 的不同行分别计分。问题和选项从测试集取，移除媒体占位符；图片/视频不传给 judge。完整预测交由 Qwen 判断，不做 regex 答案提取或本地规则判分。

## 判分规则

- MCQ：Qwen 根据问题、选项、GT 与完整预测判断最终选择，含明确语义改写，返回 0/1。
- OPEN：Qwen 判断答案语义是否相同，返回 0/1。
- MRA：Qwen 抽取 GT 和预测的主要数值，不换单位/计算；Python 使用阈值 0.50、0.55、…、0.95，计算 `mean(relative_error < 1 - threshold)`，其中 `relative_error=abs(pred-gt)/abs(gt)`。沿用参考约定，`abs(gt)<1e-9` 得 0。

模型严格返回 JSON `status/score/gt/pred`；不从 Markdown 中提取 JSON。协议错误默认修复一次，失败及 judge 的 invalid 都记 0 并留在分母，解析失败单独统计。模型调用、样本对齐或上下文出错会中止运行。

指标报告 MCQ/OPEN accuracy、MRA mean relative accuracy、按样本数加权的 `combined_flat_rawacc`，以及 source dataset、`_meta.task_type` 分组。task_type 是细分标签，不等于六个目录名。

## 启动与默认值

| 项目 | 默认值 |
| --- | --- |
| Judge | /home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct |
| Shell 测试集 | cosine_motion_test |
| Shell GPU | GPU_IDS / CUDA_VISIBLE_DEVICES；默认物理卡 1 |
| tensor parallel / batch | 1 / 4 |
| dtype / GPU memory utilization | bfloat16 / 0.8 |
| max context / max generated tokens | 8192 / 256 |
| 生成 | greedy（temperature=0、top_p=1、n=1），seed=42 |
| JSON 格式修复 | 1 次 |

脚本默认使用 `/home/linruikang/data/envs/Qwen3VL-vLLM/bin/python`（`Qwen3VL-vLLM` 环境），也可用 `EVAL_PYTHON` 显式指定解释器。环境需有 vLLM >= 0.11.0 和支持 Qwen3-VL 的 Transformers；权重必须本地可读，离线加载，不下载模型、不修改环境。默认模型 4B；参考实现用 8B，跨模型结果属于不同评测配置。

Python 入口检查 GPU 后才导入 Transformers/vLLM：必须显式选择物理 GPU 1–3 或完整 UUID；核对 nvidia-smi 身份与 PCI 序，并将物理设备限制传递给 worker。GPU 0 永远禁止，失败则停止，不自动回退。tensor parallel 数必须等于所选卡数。对已完整的 resume 仅重建汇总时不启动 GPU。

示例（占位输入路径；本轮未执行）：

```bash
cd /home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT
GPU_IDS=1 INPUT_FILE=/absolute/path/to/motion.jsonl \
OUTPUT_DIR=/home/linruikang/data/results/CL-Methods/Qwen3-VL/LoRA-FT/legacy-eval/COSINE/cosine_motion_test/run-001 \
bash evaluation/COSINE/eval_cosine.sh
```

换任务用 `DATASET=cosine_pattern_test` 等环境变量；多卡时同时设置 `TENSOR_PARALLEL_SIZE`。
输入长度加输出预算超出上下文时明确报错，不自动截断。

## 结果与续跑

默认输出放在 `/home/linruikang/data/results/CL-Methods/Qwen3-VL/LoRA-FT/legacy-eval/COSINE/<dataset>/<timestamp>`。目录名使用 dataset 名和启动时间戳；此独立入口的结果与高级 pipeline 的 `<experiment_id>/<run_id>/<job_id>/` 目录分开。

- `judge.raw.jsonl`：完整输入、所有 judge 对话/回复、解析情况、运行配置和物理 GPU 身份；逐条 flush/fsync，可作为续跑依据。
- `judge.scored.jsonl`：全量完成后原子写入的逐样本排序结果。
- `judge.metrics.json`：全量完成后原子写入的汇总与分组指标。

默认拒绝覆盖产物。`--overwrite` 只删除上述三个文件，不清理整个目录；不接受它们为软链接。
续跑需使用相同 `OUTPUT_DIR`、输入和配置并附 `--resume`。指纹核对输入/标注 SHA256、模型配置文件 SHA256、权重文件名/大小/mtime、源码哈希、依赖版本、生成配置及每条样本/回复。权重文件不计算完整内容哈希；若内容替换但大小与 mtime 保持不变，指纹无法识别。允许在物理 1–3 卡间切换，raw 每行记实际 GPU。完整结果续跑只重建 scored 与 metrics，不加载模型。截断/损坏的 raw 行会报错，需人工检查；未完成样本的中间修复尝试会重跑。
