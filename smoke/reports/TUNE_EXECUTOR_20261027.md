# 低级 LoRA 训练执行器验证（2026-10-27）

## 结论

执行器验证通过。CPU 测试 28 项全部通过；真实 Qwen3-VL-4B-Instruct / DeepSpeed ZeRO-3 smoke 中，新建 LoRA、继承前一任务 adapter 并重置训练状态、显式恢复完整 checkpoint 三条路径均正常完成。训练产物登记和内容指纹核验通过。仅使用 885 物理 GPU 1（UUID `GPU-63a25b73-5c01-919f-188a-29c116904574`）；物理 GPU 0 未使用。未修改 smoke 以外的文件，未安装依赖或执行 Git 写操作。

## 目标与范围

- 代码分支：`test/pipeline-automation`，HEAD `87dd41b08c1f936dc4883393fd846bd8c2b3ad51`，包含执行器实现 `31fea874b2e7cfaa6621e048f309b0c2b47063d7`。
- 环境：既有 `Qwen3VL-LoRA-FT`，真实训练使用本地 4B Instruct 权重和 DeepSpeed ZeRO-3。
- 数据：从 COSINE motion/hypo 训练集各选 4 条，复制标注至 smoke 外部结果目录，包含单图、多图和视频。原始标注 SHA-256 前后未变；具体样本 ID、索引和来源哈希见 `gpu/samples.json`。
- 测试入口：`smoke/scripts/test_tune_executor.py`、`smoke/scripts/run_tune_executor_gpu.py`；证据根目录为 `smoke/results/tune-executor-20261027-Eav9WyRB/`。
- 本轮仅在 `smoke/` 下新增验证内容；原先未提交的 4B shell 脚本修改、Python 缓存及 W&B 产物保留。

## CPU 测试

最终 28 项通过、0 失败、0 错误。测试阻止导入 torch、Transformers、PEFT、W&B、DeepSpeed 和 vLLM，GPU 可见设备为空。覆盖入口/依赖拒绝、job 锁、stage slot 释放、GPU 组索引边界、设备环境隔离、输入/依赖来源和指纹校验、adapter 配置兼容、adapter 继承与恢复模式、原生/DeepSpeed checkpoint 完整性、失败后 checkpoint 保留、产物登记以及本次源码语法编译。完整结果见 `cpu-v2/tests.json`。首次合成 fixture 的 DeepSpeed 配置不满足配置校验，随后只修复 smoke fixture 并在 `cpu-v2` 新目录完成验证；初次日志仍保留。

## 真实 GPU smoke

三个 job 各使用 4 条样本，每卡 batch 1、梯度累积 2、LoRA r=8/alpha=16、1 epoch，各目标 2 optimizer steps。全部使用物理 GPU 1 UUID 限制；W&B offline 设备过滤为 `[1]`，三个 job 使用独立 run ID。执行耗时分别为 106.49、107.32、96.14 秒。

| 路径 | 结果 |
| --- | --- |
| 新建 motion LoRA | 返回 0；完成 2 steps；5,898,240 个可训练参数；登记 final adapter、checkpoint-1、checkpoint-2 和 training summary。 |
| hypo 继承 motion adapter | 返回 0；使用 `--adapter_name_or_path`，不含 `--resume_checkpoint_path`；新任务重新从 step 1 开始；更新后的 adapter 不同于父 adapter。 |
| motion 完整恢复 | 从显式 checkpoint-1 恢复到 step 2，返回 0；使用 `--resume_checkpoint_path`；新目录登记 checkpoint-2、adapter 和 summary。 |

只读审计比较 288 个 adapter 张量：恢复的最终 adapter 与不中断 motion 训练逐张量完全相等，Trainer log_history 也相等。继承路径首个保存 step 与父 adapter 相等（第一 scheduler 步学习率为 0），第二步后权重不同。语言 LoRA 张量均有限，144 个 LoRA-B 张量均非零。loss 与梯度范数有限，梯度范数为正；恢复路径第二步的 loss、梯度范数和学习率与不中断训练一致。

独立审计结果为 `gpu/outcomes.verified.json`（3 项用例、11 项跨用例检查全部通过）。审计时屏蔽全部 GPU，safetensors 在 CPU 加载，确认没有初始化 CUDA。所有已记录 worker/launcher PID 均退出，GPU 1 最终为 2 MiB、0% 利用率，无残留训练进程。

训练无 OOM 或 traceback。日志有 gradient-checkpoint 的 `None of the inputs have requires_grad=True` 提示；实际梯度和权重更新已核实有效。另有既有 Linux 5.4 内核版本提示，本轮没有修改内核或环境。

## 首次收尾断言修正

GPU runner 首次在三条训练全部返回 0 后，因 smoke 断言要求恢复目录也存在 checkpoint-1 而退出。该目录实际从源 checkpoint-1 继续，仅新建 checkpoint-2，符合预期。只修正 smoke 断言，并增加 `--verify-only` 对已有结果做独立审计，未重跑任何训练。初始日志 `gpu.log` 和原 runner `gpu-runner.initial.py` 保留。不得将初始总控退出码写成 0；最终通过以独立审计结果为依据。

## 范围核验与限制

范围审计对比启动前捕获的 199 个非 smoke 文件，10 项检查全部通过：HEAD、分支、refs、tracked diff、暂存 diff、index 哈希、原有非 smoke 状态、smoke 软链接、全部非 smoke 文件内容/权限/链接和 git diff --check。非 smoke 文件变化数 0。初次审计为 `scope-report.json`；README 与报告写入后另保存 `scope-report-final.json`。未暂存、commit、push、merge 或切换分支。

本次真实训练仅覆盖 4B、单进程 ZeRO-3、每任务 4 条样本和短训练，不代表六任务完整实验、多 GPU 并行、长训练、全部媒体形态、W&B 云端同步或模型质量已验证。CPU 合成产物只用于执行器控制逻辑验收。

## 复现入口

在仓库根目录预先创建全新结果目录后，使用现有环境的 Python：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/test_tune_executor.py --result-dir "$cpu_result_dir"
```

只有明确安排实际训练时才运行 GPU 入口；它固定物理 GPU 1，配置和所选数据在新结果目录保存：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/run_tune_executor_gpu.py --result-dir "$gpu_result_dir"
```

本轮对既有训练结果的审计命令：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/run_tune_executor_gpu.py --result-dir smoke/results/tune-executor-20261027-Eav9WyRB/gpu --verify-only
```

证据不覆盖：CPU/GPU 新训练拒绝复用既有目录；verify-only 也拒绝覆盖已有 outcomes.verified.json。以上本轮训练和审计均已完成，不需重复执行。
