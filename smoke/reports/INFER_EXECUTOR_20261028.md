# 按 Adapter 的 vLLM 推理执行器验证

日期：2026-10-28（Asia/Shanghai）。885 仓库：/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT。分支：test/pipeline-automation。HEAD：483711cdccac6714ec56d5d11777f48c1739edbc；功能提交：abbedcce8913c33e98ea5eb3aab40e26f54efb87。

## 结论

第六步低级 infer 验证完成。CPU 136 项全部通过：infer executor/worker 29 项、锁与 SIGTERM 清理 4 项、foundation 回归 77 项、冻结 plan constraints 回归 26 项；0 failure、0 error。GPU 基础产物审计 81 项和增强只读审计 44 项全部通过，范围审计 10 项全部通过。

仅物理 GPU 1 上运行两个 4B Instruct job：Zero-shot/base 与 Multitask/LoRA。每 job 对 motion、hypo、pattern、nav、view、geo 六个 task 各推理 4 条，合计 48/48 非空预测，两个 CLI 均退出 0。每 job 恰好初始化一次 engine、加载一次模型，共用一个 vLLM engine 跑完六个 task。

## 配置与样本

- CPU 环境：/home/linruikang/data/envs/Qwen3VL-LoRA-FT，Python 3.11.6、PyYAML 6.0.3。
- 推理环境：/home/linruikang/data/envs/Qwen3VL-vLLM-Verify-20261025，vLLM 0.11.2。
- Base/processor：/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct。
- batch_size=4、max_model_len=32768、max_new_tokens=32；其余来自冻结 smoke 配置。
- GPU 仅物理 1，UUID GPU-63a25b73-5c01-919f-188a-29c116904574。GPU 0 未使用，2/3 未干预。
- 使用 COSINE 各 task 的真实 test 样本，覆盖单图、多图和视频；geo 此次仅包含单图。选取索引与原标注哈希见 gpu-real/samples.json。

Multitask 复用此前成功真实 whole 训练的 r8/alpha16 adapter；新建 smoke fixture 记录只供验证绑定、来源审计与推理，没有再次训练。adoption-evidence.json 明确标记 smoke_fixture=true、training_executed=false。源 manifest：

```text
smoke/results/tune-orchestrator-20261027-tfhEWVk4/gpu-real/records_root/smoke-multitask/stage-runs/tune/tune-20261027T110730Z-9e4bbbb1/jobs/tune-whole/artifacts/adapter.json
```

源 adapter_model.safetensors SHA-256：29fc4abb3b00ce472944326c32df61a8c3f26a2a523026a9dcf3e4d50be37d80。只读审计确认权重及来源指纹未变。本轮不构成新 tune→infer 端到端训练验证。

## 结果与清理

| Job | CLI 用时 | Worker 用时 | 预测 | Engine/模型加载 | 整卡显存观测最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot/base | 231.45 秒 | 85.93 秒 | 24/24 | 各 1 次 | 18,484 MiB |
| Multitask/LoRA | 308.67 秒 | 161.00 秒 | 24/24 | 各 1 次 | 18,708 MiB |

显存为约每两秒采样的整卡观测值，不是严格峰值或单进程精确峰值；共 261 次成功观测，无监控错误。两个 job 各登记六份逐 task prediction manifest 和一份 inference-summary；检查预测数量、顺序、ID、GT、完整元数据、generation、输入指纹、收据、状态及资源清理。低级 job 完成后父 infer run 保持 running，符合后续高级编排负责收尾的设计。

八个记录进程全部退出，runner 也已退出；两个临时模型视图均删除，GPU 1 最终无计算进程、显存 2 MiB。两个日志均有 NCCL destroy_process_group() 退出告警，但 CLI 返回 0、engine_shutdown=true，产物与进程清理审计通过。没有 Traceback、CUDA OOM 或 EngineCore failure；不隐去退出告警。

## 运行入口

本次真实入口如下，已经执行完成，不要重跑已有目录：

```bash
cd /home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 \
  /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B \
  smoke/scripts/run_infer_executor_gpu.py \
  --result-root smoke/results/infer-executor-20261028-DbHyQYMS/gpu-real
```

CPU 新入口为 smoke/scripts/test_infer_executor.py 和 test_infer_resources.py，参数为 --result-root <全新目录>，使用 CUDA_VISIBLE_DEVICES=、PYTHONDONTWRITEBYTECODE=1、GIT_OPTIONAL_LOCKS=0。GPU runner 的 --audit-only 只审计已有结果，不重新推理；增强审计入口 audit_infer_executor.py 同样只读核验，不加载模型。CPU 锁测试已使用 fixture 私有锁目录，不占实际 GPU 锁。

## 发现的问题

未发现需要改生产代码的缺陷。cpu-initial/ 与 cpu-corrected/ 两次初始 CPU 尝试各有 12 errors，原因是 smoke Zero-shot fixture 错误启用 tune；只修 smoke fixture 后，cpu-final/ 的 29 项和其余三套测试均通过，旧失败证据保留。提前执行增强审计时第二个 CLI 尚未退出，缺少 multitask-exit.json，曾报 FileNotFoundError；等待原 runner 完成后只重做只读增强审计，44 项通过，没有重复推理。

## 范围与证据

范围基线中的 204 个非 smoke 文件内容、权限、链接保持。HEAD/branch/refs、原 diff、暂存 diff、index SHA-256、原非 smoke status、smoke 软链接、git diff --check 共 10 项范围检查全部通过，changed_non_smoke_files 为空。原训练脚本修改、缓存与 W&B 文件保留。本轮远端改动仅 smoke README、四个验证脚本和本报告；不改生产源码、正式 YAML、环境、原数据或权重，不暂存、commit/push/merge/switch。

证据根目录：smoke/results/infer-executor-20261028-DbHyQYMS/（由既有软链接存储于外部 smoke/results）。关键文件：

- cpu-final/summary.json、cpu-resources/summary.json、cpu-foundation/tests.json、cpu-plan/tests.json。
- gpu-real/cases.json、samples.json、*-exit.json、*-gpu.jsonl、live-process-identity.jsonl。
- gpu-real/verification.json（81 项）、additional-verification.json（44 项）。
- scope-baseline.json、scope-after.json（文档写入前）、scope-final.json（收尾范围审计）。
- verification-summary.json（最终汇总）。

## 边界

仅验证 4B 单卡小样本低级推理；未验证完整数据、真实多卡、模型质量、所有 setting 的真实推理、高级 infer 编排或 eval。max_new_tokens=32 是 smoke 配置，不能据此评价答案质量。其他 setting 的 plan 约束由 CPU 回归覆盖。本轮已完成，不自动重跑或提交。
