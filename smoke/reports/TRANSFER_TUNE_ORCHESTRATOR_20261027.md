# 两轮迁移高级 tune 编排验证（2026-10-27）

## 结论与范围

第五步两轮迁移高级 tune 编排在本次范围内验证通过，未发现需要修复生产代码的问题。CPU 62 项全部通过；4B 的 6 个第一轮和 30 个第二轮真实短训练全部成功；741 项产物、参数继承、训练状态与退出审计全部通过。

- 仓库：885:/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT。
- 分支：test/pipeline-automation；HEAD da4ceb0a4ec9344d22b434392b86ffc3ebc70eda，功能提交 e8e1d3b163e31796f4446bcb65f1ccf0a95091df。
- GPU：仅物理 GPU 1，UUID GPU-63a25b73-5c01-919f-188a-29c116904574。物理 GPU 0 未使用，GPU 2/3 未干预。
- 证据根：smoke/results/transfer-tune-orchestrator-20261027-8ZYhtwJx/；外部路径 /home/linruikang/data/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/results/transfer-tune-orchestrator-20261027-8ZYhtwJx。
- 本轮只修改远端 smoke 验证脚本、README 和本报告；未改生产源码、正式 YAML、环境、原数据或 base 权重，未暂存、commit、push、merge 或切换分支。

## CPU 验证

| 套件 | 通过数 | failures / errors | 耗时（秒） |
| --- | ---: | --- | ---: |
| Transfer 编排 | 17 | 0 / 0 | 693.421 |
| 既有 Multitask/Singletask 编排回归 | 27 | 0 / 0 | 268.462 |
| CL 编排回归 | 18 | 0 / 0 | 214.437 |

三个隔离套件均退出 0，总计 62 项通过。CPU 入口屏蔽 CUDA，使用真实配置、计划、records 和高低级编排器，替换训练 worker 与 GPU 身份查询，不加载模型。

两种 Transfer setting 均验证完整 36-job 训练图：六个从同一 base 独立新建的第一轮 source，以及 source≠target 的 30 个第二轮 job。第二轮精确读取本 run 对应第一轮 final adapter，兄弟分支不串联；仅继承参数，重新初始化训练状态。检查输出 slot、manifest、producer、fingerprint、lineage 与数据绑定。

同时核对两种 setting 各 30 个 infer 计划及一个包含 30 对评分输入的 eval 计划：Transfer-round1 测 source task，Transfer-round2 测第二轮 target task。这里只验证下游计划，没有执行 infer 或 judge。

调度与故障路径覆盖父失败仅阻断其五个子 job、子失败不阻断兄弟、fail_fast、损坏父产物、成功前缀接管不重跑、两个模拟 GPU 组并发且不重叠占用、第二轮只等待各自父任务而无全局第一轮 barrier，以及计划、训练策略和精确父绑定篡改拒绝。两种 setting 的 tune 图一致性通过。

## smoke fixture 问题与处理

初次 CPU fixture 沿用生产全局设备锁，与并行运行的真实训练争用锁。仅在 smoke fixture 中将 /tmp/qwen3vl-lora-tune-locks/ 映射至各测试 case 的 cpu-resource-locks/，保留真实 flock 语义；中断精确记录的 CPU 测试及 fake worker 后完成隔离重跑。原日志、初始脚本副本及中断 PID 清单均保留。隔离入口首次因结果子目录未创建而在测试前失败，补齐目录后完成上述成功验证，失败日志也保留。

同步既有 smoke 断言：Transfer 已支持，不再断言其应被拒绝，保留 zero_shot 不支持检查；更新父绑定错误文案断言。没有为这些 smoke 问题修改生产代码或重启真实 GPU 训练。

## 真实 36-job 短训练

Python 为 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python；Python 3.11.6、torch 2.6.0+cu124、Transformers 4.57.3、PEFT 0.17.1、DeepSpeed 0.17.1、W&B 0.31.0、safetensors 0.8.0、Accelerate 1.7.0、PyYAML 6.0.3。环境未修改。

基础模型 Qwen3-VL-4B-Instruct，单进程 ZeRO-3，GPU 组 [[1]]、max_parallel_jobs=1、端口 29887。真实执行 transfer_round1 全部 36 个 job。transfer_round2 的 tune 图相同，本轮不重复真实训练，由 CPU 完整编排覆盖其 setting 分支和下游测试 task 区别。

每 task 从 COSINE train 选取并复制 4 条标注到 smoke 结果树，原媒体只读加载。motion、hypo、pattern、nav 包含单图、多图、视频；view、geo 本轮样本无视频。样本索引、ID、媒体数量与源/副本哈希见 gpu-real/samples.json；六份原始 train.json 和样本副本在审计时哈希均保持。

每 job 为 1 epoch、batch1、accumulation2、lr5e-4、LoRA r8/alpha16，共 5,898,240 个可训练参数；两次 optimizer step，每步保存 checkpoint-1/2。首步实际学习率为 0，第二步为 5e-4。其余有效参数继承默认 YAML，冻结于 transfer_round1-config.snapshot.json，未改正式默认值。W&B offline，各 job 独立 run ID 和输出目录。

训练 receipt 开始：2026-10-27T22:26:01.211685+08:00；结束：2026-10-27T23:43:04.986767+08:00。高级 CLI 进程记录耗时 4634.62 秒（约 77.24 分钟），退出码 0；最终 tune 为 succeeded，infer/eval 保持 planned，experiment 仍 active。

| 轮次 | task / source → target | 退出码 | steps | job 耗时（秒） | 两步 loss |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | motion | 0 | 2 | 106.65 | 6.3938 / 3.9938 |
| 1 | hypo | 0 | 2 | 128.15 | 6.8953 / 11.1785 |
| 1 | pattern | 0 | 2 | 171.19 | 8.1368 / 11.8147 |
| 1 | nav | 0 | 2 | 144.91 | 7.3048 / 8.9041 |
| 1 | view | 0 | 2 | 143.16 | 8.49 / 1.152 |
| 1 | geo | 0 | 2 | 152.10 | 5.8719 / 8.4064 |
| 2 | motion → hypo | 0 | 2 | 161.34 | 6.19 / 10.7275 |
| 2 | motion → pattern | 0 | 2 | 139.30 | 7.524 / 11.6196 |
| 2 | motion → nav | 0 | 2 | 146.86 | 6.3657 / 7.8672 |
| 2 | motion → view | 0 | 2 | 158.80 | 8.0557 / 1.1304 |
| 2 | motion → geo | 0 | 2 | 146.57 | 5.1799 / 6.4895 |
| 2 | hypo → motion | 0 | 2 | 151.64 | 5.3452 / 3.9209 |
| 2 | hypo → pattern | 0 | 2 | 158.82 | 7.0802 / 10.2441 |
| 2 | hypo → nav | 0 | 2 | 145.80 | 6.0027 / 7.9415 |
| 2 | hypo → view | 0 | 2 | 105.50 | 7.3102 / 1.1115 |
| 2 | hypo → geo | 0 | 2 | 106.51 | 4.9817 / 7.3784 |
| 2 | pattern → motion | 0 | 2 | 107.60 | 5.6368 / 4.1538 |
| 2 | pattern → hypo | 0 | 2 | 109.27 | 6.0368 / 9.8909 |
| 2 | pattern → nav | 0 | 2 | 153.49 | 6.3109 / 8.1869 |
| 2 | pattern → view | 0 | 2 | 147.09 | 7.173 / 1.1079 |
| 2 | pattern → geo | 0 | 2 | 154.65 | 5.2485 / 7.9946 |
| 2 | nav → motion | 0 | 2 | 152.10 | 5.4266 / 3.4311 |
| 2 | nav → hypo | 0 | 2 | 109.94 | 5.9316 / 10.2132 |
| 2 | nav → pattern | 0 | 2 | 105.60 | 7.2715 / 10.9877 |
| 2 | nav → view | 0 | 2 | 106.06 | 7.4515 / 1.0921 |
| 2 | nav → geo | 0 | 2 | 105.85 | 4.7878 / 4.4419 |
| 2 | view → motion | 0 | 2 | 108.96 | 5.9085 / 4.1155 |
| 2 | view → hypo | 0 | 2 | 107.55 | 6.359 / 10.5137 |
| 2 | view → pattern | 0 | 2 | 105.79 | 7.4892 / 10.8503 |
| 2 | view → nav | 0 | 2 | 108.07 | 6.5766 / 8.3733 |
| 2 | view → geo | 0 | 2 | 106.10 | 5.363 / 8.2254 |
| 2 | geo → motion | 0 | 2 | 108.39 | 5.8106 / 3.1126 |
| 2 | geo → hypo | 0 | 2 | 108.74 | 6.2199 / 10.1577 |
| 2 | geo → pattern | 0 | 2 | 105.87 | 7.1558 / 11.359 |
| 2 | geo → nav | 0 | 2 | 108.76 | 6.2255 / 6.0906 |
| 2 | geo → view | 0 | 2 | 106.45 | 7.6657 / 1.1055 |

## 参数继承与状态重置

36 个 final adapter 均包含 288 个有限的语言 LoRA 张量，其中 144 个 LoRA-B 全部非零。所有 job 的两步梯度范数非零且有限，loss 有限。首轮六个 job 的 checkpoint-1 LoRA-B 全零；第二轮全部 30 个 job 的 checkpoint-1 所有 288 个张量与各自 source 第一轮 final adapter 精确相等。首步学习率为零，因此可以直接核验初始参数继承；第二步后 36 个 final adapter 均有权重更新。

每个 job 的 checkpoint-1/2 分别核对 optimizer step=1/2、DeepSpeed global_steps=1/2、global_samples=2/4、scheduler last_epoch=1/2、_step_count=2/3、trainer global_step=1/2、epoch=0.5/1.0；保存后的 scheduler last_lr=5e-4/0。第二轮未继续父任务的 optimizer、scheduler 或训练进度，未传 resume_checkpoint_path。

全部第二轮 job 的 parent slot、producer、fingerprint、命令行 adapter 路径和 inherited_inputs 均绑定 source 第一轮 final，而非兄弟 job。36 份 final adapter、72 份 step checkpoint、36 份 training summary 均通过 manifest 内容审计；36 个输出目录和 W&B run ID 相互独立。六个第一轮 adapter 在所有后续派生完成后仍保持可校验。CUDA 可见 UUID 和 W&B GPU selector 均仅对应物理 GPU 1。

CPU --verify-only 返回 0、未初始化 CUDA，outcomes.verified.json 的 741 项全部通过。565 个训练启动器、worker 及 GPU 观测记录 PID 在审计时均已退出；runner 亦正常退出。训练后 GPU 1 无计算进程、显存 2 MiB。每 5 秒监控共 855 条有效记录、0 次查询错误，整卡观测显存最大 18,636 MiB，是采样最大值，并非严格峰值。训练日志无 traceback 或 CUDA OOM。

GPU runner 从启动到最终审计未修改，当前文件与 runner.launch-version.py 的 SHA-256 均为 86d836bcb8b0a5c4ddda131e87b3eded8f8f9f94e49925c3d6916db39769872b。未重复训练。

## 范围核验与证据

scope-after.json 的 10 项检查全部通过：201 个非 smoke 文件的哈希/权限/链接、HEAD/分支/refs、index 二进制哈希、暂存内容、原非 smoke 工作区 status/diff、smoke/data 与 smoke/results 链接均保持，git diff --check 通过。完成本文与 README 后的最终检查记录为 scope-after-final.json；完成汇总为 verification-summary.json。

远端仅六个验证文件变化：smoke/README.md、smoke/scripts/test_tune_orchestrator.py、smoke/scripts/test_cl_tune_orchestrator.py、smoke/scripts/test_transfer_tune_orchestrator.py、smoke/scripts/run_transfer_tune_orchestrator_gpu.py、smoke/reports/TRANSFER_TUNE_ORCHESTRATOR_20261027.md。数据、配置快照、完整日志、权重和 JSON 证据保存在忽略的 results 树。

主要证据：transfer-cpu-isolated/tests.json、regression-cpu-isolated/tests.json、cl-cpu-isolated/tests.json、对应成功日志、cpu-initial-interruption.json、cpu-initial-final-stop.json、environment-and-scripts.json、gpu-runner.log、gpu-real/transfer_round1-command.json、transfer_round1-config.snapshot.json、transfer_round1-exit.json、transfer_round1-gpu.jsonl、samples.json、outcomes.verified.json、verify-only.log、records_root/、gpu-before.json、gpu-after-training.json 和各次 scope 报告。

## 入口与边界

CPU 入口为 smoke/scripts/test_transfer_tune_orchestrator.py，另回归 test_tune_orchestrator.py、test_cl_tune_orchestrator.py。均须预先创建全新 smoke/results 子目录并传 --result-dir；设置 CUDA_VISIBLE_DEVICES=、PYTHONDONTWRITEBYTECODE=1、GIT_OPTIONAL_LOCKS=0。

真实训练入口为 smoke/scripts/run_transfer_tune_orchestrator_gpu.py --result-dir <全新目录>，固定物理 GPU 1，会执行全部 36 个短 job。只读审计使用相同结果目录并增加 --verify-only，必须屏蔽 CUDA；拒绝覆盖已有 outcomes.verified.json。上述训练和审计均已完成，不重复启动。

通过范围仅为 4B 单卡、每 task 4 条样本、每 job 两步训练。真实多卡并发、完整数据/长训练、模型质量、W&B 云端显示及实际 infer/eval 编排未验证；故障注入和 Transfer-round2 运行分支为 CPU 模拟。不自动提交或合并，后续按用户指示。
