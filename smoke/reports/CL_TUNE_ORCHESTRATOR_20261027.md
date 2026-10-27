# 持续学习高级 tune 编排验证（2026-10-27）

## 结论与范围

第四步 CL 高级 tune 编排在本次验证范围内通过，未发现需要修复生产代码的问题。CPU 新增 18 项与既有回归 27 项共 45 项全部通过；真实 4B 六阶段短训练全部返回 0；135 项产物、权重与退出审计全部通过。

- 仓库：885:/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT。
- 分支：test/pipeline-automation；HEAD ee61f5d7a52e6eb7fc40fb25964f94544991087b，功能提交 952768b4ffd53c0518e88cc2224294445e34cde1。
- GPU：仅物理 GPU 1，UUID GPU-63a25b73-5c01-919f-188a-29c116904574；物理 GPU 0 未使用，GPU 2/3 未干预。
- 证据根：smoke/results/cl-tune-orchestrator-20261027-lco2v7cl/；外部路径 /home/linruikang/data/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/results/cl-tune-orchestrator-20261027-lco2v7cl。
- 只新增或修改 smoke 验证脚本、README 和本报告；未修改生产代码、正式 YAML、环境、原数据或 base 权重，未暂存、commit、push、merge 或切换分支。

## CPU 编排检查

cl-cpu/tests.json：18 项、0 failure、0 error，耗时 193.524 秒。regression-cpu/tests.json：27 项、0 failure、0 error，耗时 220.952 秒。真实使用配置、冻结计划、records、高低级执行器，替换训练 worker 与 GPU 身份查询，不导入模型或使用 GPU。

新增覆盖默认 motion → hypo → pattern → nav → view → geo、完整逆序自定义顺序、精确父 manifest/producer/slot/fingerprint/lineage、FRESH/CONTINUE 策略、严格串行调度，以及新建/继承与显式 checkpoint 状态恢复的区别。即使提供两个模拟 GPU 组并允许并行，CL 实测同时运行数仍为 1。

失败路径包括首阶段失败、中段失败且 fail_fast=false、父 adapter 损坏、清理失败、异常退出、子进程启动失败，均阻断后续依赖；成功前缀接管保留已有 execution receipt，不重复启动成功 job。计划顺序、lineage、训练策略、父绑定或注册被篡改均拒绝；仅 tune 执行，infer/eval 保持 planned。

本轮为既有通用 CPU fixture 增加父 adapter 的 --input 支持，并删除其“CL 尚不支持”的过时断言；其他不支持 setting 的检查保留。CPU 模拟设备并发与故障注入不代表真实多卡或 GPU 故障恢复验证。

## 真实六阶段短训练

Python 为 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python；Python 3.11.6，torch 2.6.0+cu124、Transformers 4.57.3、PEFT 0.17.1、DeepSpeed 0.17.1、W&B 0.31.0、safetensors 0.8.0、Accelerate 1.7.0。环境未修改。

基础模型 Qwen3-VL-4B-Instruct，每 task 从原 COSINE train 取 4 条并复制标注到证据目录，样本索引、ID、媒体数量和源/副本哈希见 gpu-real/samples.json。motion、hypo、pattern、nav 覆盖单图、多图和视频；view、geo 的本轮样本无视频。媒体仍从原数据位置只读加载。

单进程 ZeRO-3，GPU 组 [[1]]，端口 29883；1 epoch、batch 1、accumulation 2、主 lr=5e-4、LoRA r8/alpha16，共 5,898,240 个可训练参数，每阶段 2 optimizer steps，每步保存 checkpoint。首步实际学习率为 0，第二步为 5e-4；完整有效参数及媒体采样来自 gpu-real/cl-config.snapshot.json，未修改默认 YAML。W&B offline，每阶段独立 run ID 与输出目录。

| 阶段 | 退出码 | steps | job 耗时（秒） | 两步 loss |
| --- | ---: | ---: | ---: | --- |
| motion | 0 | 2 | 106.45 | 6.3938 / 3.9938 |
| hypo | 0 | 2 | 107.70 | 6.1935 / 10.7670 |
| pattern | 0 | 2 | 106.14 | 5.5104 / 8.6246 |
| nav | 0 | 2 | 107.11 | 2.4923 / 4.2105 |
| view | 0 | 2 | 105.30 | 1.1339 / 0.8874 |
| geo | 0 | 2 | 105.12 | 0.5665 / 0.6664 |

高级 CLI 总耗时 645.49 秒，退出码 0。每 5 秒采样 GPU 1，共 128 条有效观测；整卡显存观测最大值 12,330 MiB，并非严格峰值。

## 参数继承与状态重置证据

六个 final adapter 均为 288 个有限的语言 LoRA 张量，144 个 LoRA-B 最终全部非零。第一阶段 checkpoint-1 的 LoRA-B 全零；后五阶段 checkpoint-1 的全部 288 张量均与各自父阶段 final adapter 精确相等。由于首步学习率为零，该比较直接核验初始参数继承；第二步后六阶段权重均更新。

每个阶段分别检查 checkpoint-1 和 checkpoint-2：optimizer step 为 1/2，DeepSpeed global_steps 为 1/2、global_samples 为 2/4，scheduler last_epoch 为 1/2、_step_count 为 2/3，trainer global_step 为 1/2、epoch 为 0.5/1.0。阶段间未携带上阶段训练进度，各阶段重新启动 optimizer、scheduler 及 epoch/global step；未传入 resume_checkpoint_path。继承参数的各次 manifest 与父输出完全一致，lineage、producer、slot、fingerprint 和 inherited_inputs 对齐。

每个阶段登记 final adapter、checkpoint-1、checkpoint-2、training summary，manifest 内容指纹有效。高级 tune 为 succeeded，infer/eval 保持 planned，experiment 仍 active；本轮未执行推理或评测。CUDA UUID 和 W&B GPU selector 均只指向物理 GPU 1。

CPU --verify-only 审计未初始化 CUDA，gpu-real/outcomes.verified.json 的 135 项检查全部通过；全部 19 个记录 PID 均退出，训练后物理 GPU 1 无计算进程、显存 2 MiB。日志无 traceback 或 CUDA OOM。已有内核版本和 gradient-checkpoint 输入 requires_grad 提示仍出现，实测梯度有限且非零、adapter 已更新；未因此修改生产代码或环境。

训练启动版 smoke 脚本和哈希已保存；训练进行时仅扩充 verify() 的 optimizer/engine/scheduler 计数审计，未修改训练逻辑或重启训练。新旧 SHA 和审计版本分别见 gpu-real/environment.json、audit-script-change.json 和 run_cl_tune_orchestrator_gpu.audit-version.py。

## 范围核验与边界

scope-after.json 的 10 项检查全部通过：201 个非 smoke 文件的内容/权限/链接、HEAD/分支/refs、index 二进制哈希、暂存 diff、原非 smoke 工作区 status/diff、smoke 软链接均保持，git diff --check 通过。README 和本报告保存后的最终核验记录为 scope-after-final.json；本轮汇总为 verification-summary.json。

本轮仅涉及 smoke/README.md、scripts/test_tune_orchestrator.py、scripts/test_cl_tune_orchestrator.py、scripts/run_cl_tune_orchestrator_gpu.py 和 reports/CL_TUNE_ORCHESTRATOR_20261027.md。数据、完整日志、配置快照、权重与结果均在忽略的 results 树保存。

仅证明 4B 单卡、每 task 4 条样本与两步训练的六阶段链路；完整训练、真实多卡、自定义顺序的真实 GPU 训练、W&B 云端显示与模型质量未验证。Transfer 和 infer/eval 编排不在本轮范围。已完成的训练不重复启动，未自动提交。

## 入口与主要证据

以下命令已执行；训练入口要求全新 smoke/results 目录，不能覆盖已有证据：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/test_cl_tune_orchestrator.py --result-dir smoke/results/cl-tune-orchestrator-20261027-lco2v7cl/cl-cpu
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/test_tune_orchestrator.py --result-dir smoke/results/cl-tune-orchestrator-20261027-lco2v7cl/regression-cpu
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/run_cl_tune_orchestrator_gpu.py --result-dir smoke/results/cl-tune-orchestrator-20261027-lco2v7cl/gpu-real
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B smoke/scripts/run_cl_tune_orchestrator_gpu.py --result-dir smoke/results/cl-tune-orchestrator-20261027-lco2v7cl/gpu-real --verify-only
```

主要证据：cl-cpu/tests.json、regression-cpu/tests.json、gpu-runner.log、gpu-real/cl-command.json、cl-config.snapshot.json、cl-exit.json、cl-gpu.jsonl、samples.json、outcomes.verified.json、verify-only.log、records_root/、gpu-before.json、gpu-after-training.json 和 scope-before.json。
