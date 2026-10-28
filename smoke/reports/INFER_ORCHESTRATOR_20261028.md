# 实验高级 infer 编排验证

日期：2026-10-28（Asia/Shanghai）。885 仓库：/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT。分支 test/pipeline-automation，HEAD 9829aa01cde891432dcbc6a75b98bb01e759d848，功能提交 5f85e52cbffa9d42f9fd887feafd0d74b26e61c7。

## 验证结论

第七步高级 infer 编排验证通过。CPU 共 176 项，0 failures、0 errors；真实 GPU 产物、输入、来源与退出审计 119 项全部通过。仅在物理 GPU 1 上执行两个 4B 高级 infer run：Zero-shot/base 与 Multitask/LoRA，各只启动一个低级 job，加载一次 vLLM engine 和一次模型，处理六个 task 的全部绑定样本；合计 48/48 条非空预测。高级 stage、job、汇总与 invocation receipt 均为 succeeded；eval 保持 planned，未启动评测。

## CPU 覆盖

| 套件 | 项数 | 结果 |
| --- | ---: | --- |
| 高级 infer 编排 cpu-02/summary.json | 40 | 全通过 |
| 低级 infer 执行器 regression-executor/summary.json | 29 | 全通过 |
| 资源锁与中断 regression-resources/summary.json | 4 | 全通过 |
| 基础配置与记录 regression-foundation/tests.json | 77 | 全通过 |
| 冻结计划约束 regression-plan/tests.json | 26 | 全通过 |

高级编排 40 项耗时 668.936 秒。覆盖七种 setting 的 adapter/task 绑定、自定义 CL 顺序、Transfer-round2 全 30 jobs 模拟执行、GPU 分组和串并行、覆盖参数、失败策略、中断清理、成功 job 复核后跳过、冻结记录与产物篡改拒绝和 CLI。CPU 使用明确的合成模型与 worker，不执行真实模型或 GPU 计算。

首轮 cpu-01/ 与 cpu-01.log 保留：其中存在 smoke fixture 对 Transfer 字段的错误假定，以及单 GPU group 却设置 max_parallel_jobs=2 等配置错误；只修 smoke 测试夹具后，cpu-02 的 40 项全过。通过时源码及哈希保存在 cpu-02-source.py、cpu-02-source.sha256。未修改生产代码来适配测试。

## 真实推理配置与输入

- CPU/总控环境：/home/linruikang/data/envs/Qwen3VL-LoRA-FT。
- vLLM 环境：/home/linruikang/data/envs/Qwen3VL-vLLM-Verify-20261025。
- Base/processor：/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct。
- batch_size=4、max_model_len=32768、max_new_tokens=32；其他参数见各 run 冻结配置。
- 仅物理 GPU 1，UUID GPU-63a25b73-5c01-919f-188a-29c116904574；GPU 0 未使用，2/3 未干预。
- 六个 task 各 4 条真实 test 样本，覆盖单图、多图和视频。索引与原 annotation SHA256 见 gpu-real/samples.json。

| Task | 原 test 索引 |
| --- | --- |
| motion | 0, 405, 607, 1 |
| hypo | 0, 230, 240, 1 |
| pattern | 7, 20, 0, 1 |
| nav | 0, 85, 1, 2 |
| view | 224, 0, 797, 1 |
| geo | 0, 1, 2, 3 |

Multitask 显式收养已有成功真实 whole 训练的 r8/alpha16 adapter。adoption-evidence.json 记录 training_executed=false；本轮没有训练，不构成新 tune→infer 训练链验证。来源 adapter_model.safetensors SHA256 为 29fc4abb3b00ce472944326c32df61a8c3f26a2a523026a9dcf3e4d50be37d80，审计确认源权重与输入指纹未变。

## 真实运行结果

| Setting | 高级 CLI 退出码 | CLI 用时 | 预测 | Engine/模型加载 | 整卡显存观测最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot/base | 0 | 273.303 秒 | 24/24 非空 | 各 1 次 | 18,532 MiB |
| Multitask/LoRA | 0 | 289.436 秒 | 24/24 非空 | 各 1 次 | 18,708 MiB |

两组分别 132/140 次显存观测，所列为整卡采样最大值，非严格峰值。119 项审计核验逐 task prediction、GT/索引/ID/完整元数据、七份输出 manifest、冻结输入、adapter 来源、UUID、上层汇总和清理回执。八个记录 PID（322747、322751、329597、330224、342941、342945、349299、350274）及总控 runner 319287 均退出；GPU 1 最终无计算进程、显存 2 MiB。

没有 traceback 或 CUDA OOM。完整 verification.json 保留 110 条 warning 记录，主要为 vLLM 仅支持语言 LoRA、忽略视觉模块的提示及 tokenizer/kernel 提示，另有两次 NCCL destroy_process_group() 退出告警。两次 CLI 退出 0，engine_shutdown、产物与进程清理均通过；不隐去告警，也不将其描述为无告警运行。

## 入口与证据

新增验证脚本：smoke/scripts/test_infer_orchestrator.py、smoke/scripts/run_infer_orchestrator_gpu.py。GPU runner 复用低级 smoke fixture，但实际启动的是生产高级 scripts/experiments/run_infer.py --run-dir <冻结 run>，完整实际命令见 gpu-real/high-cases.json。gpu-real/cases.json 是底层 fixture 写下的低级入口，不是本轮真实高级执行入口。

CPU 入口参数 --result-root <全新且尚不存在的 smoke/results 子目录>，运行时设置 CUDA_VISIBLE_DEVICES=、PYTHONDONTWRITEBYTECODE=1、GIT_OPTIONAL_LOCKS=0。GPU runner 固定物理 GPU 1，--result-root <全新目录> 会启动真实推理；--audit-only 仅审计已有结果。已完成本轮，不重复启动或覆盖证据。

证据根：smoke/results/infer-orchestrator-20261028-t1qglfpo/。

- cpu-01/、cpu-01.log：首轮失败证据；cpu-02/summary.json 与 cpu-02.log：最终高级 CPU 结果。
- regression-executor/、regression-resources/、regression-foundation/、regression-plan/：回归结果。
- gpu-real/high-cases.json、samples.json、zero_shot.yaml、multitask.yaml：实际入口与独立 smoke 配置。
- gpu-real/verification.json、gpu-exits.json、*-gpu.jsonl、gpu-after.json：119 项核验、退出与显存证据。
- gpu-real/records_root/：完整冻结记录、高低级状态、输入绑定、产物和汇总。
- scope-baseline.json、scope-final.json、verification-summary.json：最终范围审计与汇总。

## 修改边界与限制

远端仅新增两个 smoke 脚本和本报告，更新 smoke/README.md；证据写入既有 smoke/results 外部链接。生产代码、正式/default YAML、环境、原数据与权重未改；原训练脚本修改、缓存/W&B 文件和链接保留。最终范围审计 10 项全过：206 个非 smoke 文件内容、权限和链接，以及 HEAD/branch/refs、原 diff/status、暂存 diff/index 字节及 smoke 链接保持，git diff --check 通过。详见 scope-final.json 及 verification-summary.json。未暂存、commit/push/merge/switch。

真实 GPU 仅证明 4B 单卡 Zero-shot 与 Multitask 小样本高级 infer 可运行；其他 setting、并发及失败路径是 CPU 模拟覆盖，未真实运行所有 adapter、多卡或完整数据。max_new_tokens=32 为 smoke 设置，非空输出不代表答案正确；未验证模型质量或 eval 编排。本轮验证完成，不自动开展后续实验或提交。
