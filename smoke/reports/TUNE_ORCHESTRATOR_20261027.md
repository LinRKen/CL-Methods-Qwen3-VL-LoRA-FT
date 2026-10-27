# Multitask 与 Singletask 高级 tune 编排验证（2026-10-27）

## 结论与范围

第三步编排在本次验证范围内通过，未发现需要修改生产代码的问题。CPU 27 项通过；真实 Qwen3-VL-4B-Instruct / 单进程 DeepSpeed ZeRO-3 中，Multitask 的一个 whole job 和 Singletask 的六个独立 task jobs 均完成。仅使用 885 物理 GPU 1（`GPU-63a25b73-5c01-919f-188a-29c116904574`），物理 GPU 0 未使用，GPU 2/3 未干预。

- 分支：`test/pipeline-automation`；HEAD `0dff0a83314efcd1e3296b3a85040ee81db9c025`，包含功能提交 `7cc1683`。
- 生产入口：`scripts/experiments/run_tune.py` 和 `lib/tune_orchestrator.py`，调用真实低级 executor 和训练代码。
- 本轮仅新增两个 smoke 验证脚本、一个报告并更新 smoke README。原有训练脚本修改、缓存、W&B 产物及软链接保留；未修改环境或生产源码，未暂存、commit、push、merge 或切换分支。index 字节差异另见下文。
- 证据根目录：`smoke/results/tune-orchestrator-20261027-tfhEWVk4/`；其外部位置为 `/home/linruikang/data/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/results/tune-orchestrator-20261027-tfhEWVk4`。

## CPU 控制逻辑

最终结果 `cpu-corrected/tests.json`：27 项、0 failure、0 error，耗时 154.973 秒。测试真实加载 config、pipeline、records、orchestrator 及低级 executor；仅替换设备身份与训练 worker，模型库导入被阻止，不使用 GPU。

覆盖单 whole / 六个独立 task 的计划和 adapter slot、参数覆盖传递、互不重叠 GPU 组并发、同 GPU 与端口冲突串行、fail-fast 与继续独立任务、worker 异常退出及清理失败、成功 job 核验后跳过、损坏产物拒绝、重复执行/记录篡改/占用锁拒绝、CLI 冻结快照与参数约束、SIGTERM 取消和子进程收尾。CPU 中的多 GPU 调度使用模拟设备，不能代替真实多卡验证。

首轮 smoke 有 2 failure 和 3 error，原因均为 fixture/断言设置：tuple 与 JSON list 比较、zero-shot fixture 错启 tune，以及并行上限超过 GPU 组数。仅修改 smoke；第二次因本地没有 `python` 且 shell 未中止而再次运行旧版本，未作为最终通过证据。最终使用修正版本在新目录运行全部通过；`cpu-initial/`、`cpu-final/`、对应日志与初始源码均保留。

## 真实高级 CLI 训练

使用既有 `/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python`；环境版本见 `environment.json`。Python 3.11.6，torch 2.6.0+cu124、Transformers 4.57.3、PEFT 0.17.1、DeepSpeed 0.17.1、W&B 0.31.0。

base 为 `/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct`。从 COSINE whole 和六个 task 的 train 各取 4 条，复制标注至证据目录；样本索引、ID、媒体数量与来源哈希见 `gpu-real/samples.json`。whole、motion、hypo、pattern、nav 含单图/多图/视频，view、geo 本轮样本无视频。图片和视频资源仍读取原数据位置，不修改原数据。

每 job 使用 1 epoch、batch 1、梯度累积 2、LoRA r8/alpha16，共 2 optimizer steps，5,898,240 个可训练参数；其他训练参数继承默认配置。W&B offline，所有 job 使用独立 run ID；W&B 设备过滤和 CUDA UUID 都限定物理 GPU 1。Multitask 端口 29881，Singletask 端口 29882；单卡串行，无自动续训。配置快照和确切命令位于 `gpu-real/`。

| Setting / task | 退出码 | steps | job 耗时（秒） | train loss |
| --- | ---: | ---: | ---: | ---: |
| multitask/tune-whole | 0 | 2 | 106.85 | 8.6397 |
| singletask/tune-motion | 0 | 2 | 107.72 | 5.1938 |
| singletask/tune-hypo | 0 | 2 | 106.36 | 9.0369 |
| singletask/tune-pattern | 0 | 2 | 104.53 | 9.9757 |
| singletask/tune-nav | 0 | 2 | 107.26 | 8.1045 |
| singletask/tune-view | 0 | 2 | 105.23 | 4.8210 |
| singletask/tune-geo | 0 | 2 | 105.78 | 7.1392 |

Multitask 高级 CLI 共耗时 108.96 秒，Singletask 高级 CLI 共耗时 643.32 秒，两者均返回 0。每 5 秒仅采样物理 GPU 1，整卡显存观测最大值分别为 12,184 和 12,330 MiB，不是严格峰值。

七个 job 的数据路径逐项绑定相应 task，输出目录、W&B run ID 独立。每个 job 均登记 final adapter、checkpoint-1、checkpoint-2 和 training summary；全部 manifest 内容指纹有效。高级 tune 状态均为 succeeded，infer/eval 保持 planned，因此 experiment 为 active，未运行推理或评测。

CPU 只读审计在屏蔽 GPU 下加载 safetensors：每个 adapter 为 288 个语言 LoRA 张量，其中 144 个 LoRA-B 最终非零，所有数值有限。七个 checkpoint-1 的初始权重逐张量相同、LoRA-B 全零（首步学习率为零），每个最终 adapter 都已更新；这与各 task 从同一 base 和相同 seed 独立初始化一致。每个 job 都未传入 adapter 继承或 checkpoint 恢复参数，global step/epoch 为新的训练状态。loss 与梯度范数有限且梯度范数非零。

独立产物审计 gpu-real/outcomes.verified.json 的 134 项检查全部通过，七个 job 均通过，审计未初始化 CUDA。全部 23 个记录 PID 均退出。最终物理 GPU 1 显存 2 MiB、利用率 0%，无计算进程；训练日志无 traceback 或 CUDA OOM。

日志中既有 gradient-checkpoint 输入 requires_grad 提示和 Linux 5.4 内核版本提示不改变上述权重与梯度实测结果；未因此修改环境。初始训练脚本版本保留为 `gpu-runner.training.py`；训练后仅增加 smoke 只读审计检查，未重跑训练。

## 范围审计与边界

收尾 scope-after.json 与报告修订后的 scope-after-final.json 均对比启动前的 201 个非 smoke 文件：9/10 项检查通过，唯一不一致是 index 二进制哈希。非 smoke 文件内容/权限/链接变化数为 0；HEAD、分支、refs、暂存 diff、原非 smoke 工作区状态及 diff、smoke 链接均保持，git diff --check 通过。smoke 本轮仅 README、两个脚本和本报告四个文件变化。

额外只读核验 index-content-audit.json 的 6 项检查全部通过：197 个暂存条目的路径、mode、object ID 与未变的 HEAD 完全相同，原始与当前 staged diff 均为空，无冲突条目或 index.lock。index 的 mtime 为 2026-10-27 19:16:31 +08:00；本轮有一次 git status 未设置 GIT_OPTIONAL_LOCKS=0，可能刷新了 stat cache。未保存原 index 二进制，不能直接证明所有字节差异的来源，因此保留此范围审计例外，不宣称 index 字节未变，也不尝试恢复或重写 index。此差异未改变任何暂存文件内容。

本轮只证明 4B、单卡、每任务 4 条样本和短训练的高级 tune 编排。真实多卡/多 job 并行、完整训练集、长训练、全量媒体类型、W&B 云端同步和模型质量尚未验证；CL/Transfer 与 infer/eval 编排不在本轮范围。

## 入口与证据

CPU 入口为 `smoke/scripts/test_tune_orchestrator.py --result-dir <全新结果目录>`，运行前设置 `CUDA_VISIBLE_DEVICES=` 和 `PYTHONDONTWRITEBYTECODE=1`。

实际训练入口为 `smoke/scripts/run_tune_orchestrator_gpu.py --result-dir <全新结果目录>`，固定物理 GPU 1，会运行全部七个短 job。本轮已经完成，不重复执行。标注/配置/权重/完整日志只在忽略的 results 目录保存。

本轮只读审计命令（已执行，不覆盖既有证据）：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/run_tune_orchestrator_gpu.py --result-dir smoke/results/tune-orchestrator-20261027-tfhEWVk4/gpu-real --verify-only
```

主要证据为 `cpu-corrected/tests.json`、`gpu-driver.log`、`gpu-real/{multitask,singletask}-exit.json`、`gpu-real/outcomes.verified.json`、`gpu-real/records_root/`、`gpu-real/*-gpu.jsonl`、`scope-before.json`、`scope-after.json`、`scope-after-final.json` 与 `index-content-audit.json`。未暂存、commit、push、merge 或切换分支。
