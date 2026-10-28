# 第八步批量 Qwen 柔性评测执行器验证

日期：2026-10-28（Asia/Shanghai）。仓库：885 /home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT。分支 test/pipeline-automation，HEAD 75145a9e2ee75aaed9a95506a1697bffc3c5820f，功能提交 fb0d3b118b8ae58f144973c698d3159f9a487140。

## 验证结论

第八步低级批量评测执行器的 CPU 覆盖、真实评测与产物核验均完成。真实运行只使用物理 GPU 1，Zero-shot/base 和 Multitask/whole-LoRA 各执行一个冻结 eval job，分别对 motion、hypo、pattern、nav、view、geo 六个 task 每项四条样本完成柔性评测，共 48 条。两 job 的 CLI 和 worker 均返回 0，各只加载一次 Qwen3-VL-8B-Instruct judge，正确关闭 engine；32 项 runner 审计及 214 项扩展审计全部通过。

CPU eval 套件覆盖 40 个用例。首轮 38 项通过，2 项因 smoke fixture 本身错误而报错；只修正测试夹具后，这两项定向复测通过。两处 fixture 错误分别为把 compute_mra 引用到错误的 legacy 模块、使用不支持的 artifact kind model；修复为独立数值预期及 base_model 类型。另有资源/中断用例 3/3、基础配置回归 77/77、计划约束回归 26/26 通过。上述首轮 fixture 错误已保留在证据目录，未通过改生产代码规避。

## CPU 覆盖与结果

CPU eval 用例覆盖 Zero-shot、Multitask、Singletask、CL 与两种 Transfer setting 的成对输入及跨 adapter 重复样本；MCQ、OPEN、MRA、judge 修复、invalid/解析失败计零和加权 combined；prediction manifest 缺失/重复/额外、跨 experiment、失败 producer、prediction/GT/plan/task 篡改、GPU 0 拒绝、输出覆盖和终态 job 拒绝、worker 中断清理、raw/scored/metrics/request 篡改、CLI 返回码和 job 锁。实际 legacy judge parsing/scoring、records、executor/worker 逻辑在 CPU 中执行；模型、vLLM 与 GPU 调用被 mock 或拦截。

## 真实运行设置

- 执行环境：/home/linruikang/data/envs/Qwen3VL-vLLM-Verify-20261025，Python 3.11.17、PyTorch 2.9.0、vLLM 0.11.2、Transformers 4.57.3、PyYAML 6.0.3。
- Judge：/home/linruikang/data/checkpoints/Qwen3-VL-8B-Instruct；dtype bfloat16、batch_size 4、tensor_parallel_size 1、gpu_memory_utilization 0.8、max_model_len 8192、max_new_tokens 256、repair_attempts 1、seed 42。
- 设备：物理 GPU 1，UUID GPU-63a25b73-5c01-919f-188a-29c116904574。GPU 0 未使用，GPU 2/3 未干预。
- 输入：复用第七步已有 Zero-shot/base 与 Multitask/whole-LoRA infer smoke 的预测 manifest 和原 annotation；本轮没有训练或重新推理。每个 task 4 条，总计每个 setting 24 条。

| Setting | 总样本 | Valid / invalid | Score sum | combined_flat_rawacc | judge loads | CLI 用时 | 整卡显存观测最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot/base | 24 | 10 / 14 | 5.9 | 24.58% | 1 | 389.99 秒 | 19,254 MiB |
| Multitask/whole-LoRA | 24 | 12 / 12 | 7.9 | 32.92% | 1 | 386.96 秒 | 19,254 MiB |

两组 judge parse failures 均为 0，engine shutdown 均成功。各 task combined-flat 分数如下：

| Task | Zero-shot/base | Multitask/whole-LoRA |
| --- | ---: | ---: |
| motion | 25.00% (1.0/4) | 25.00% (1.0/4) |
| hypo | 0.00% (0/4) | 25.00% (1.0/4) |
| pattern | 75.00% (3.0/4) | 75.00% (3.0/4) |
| nav | 25.00% (1.0/4) | 25.00% (1.0/4) |
| view | 22.50% (0.9/4) | 22.50% (0.9/4) |
| geo | 0.00% (0/4) | 25.00% (1.0/4) |

上述分数只记录流程产物，不用于比较模型效果：复用的 infer smoke 配置将 max_new_tokens 设为 32，抽查可见多条预测在句子中途截断；invalid 分别为 14/24 和 12/24，并按规则计零且保留在分母。这组小样本不代表正式模型准确率。

整卡显存采样每组约 187–188 次，所列为观测最大值而非严格峰值。两次 eval 日志都含一次 NCCL destroy_process_group() 退出告警；CLI/worker 仍返回 0，cleanup receipt 成功，记录的六个进程 PID 均退出。GPU 1 收尾无计算进程、显存 2 MiB。无 OOM 或 traceback。

## 证据与范围

证据目录：smoke/results/eval-executor-20261028-XekckS2O/。

- cpu-initial/ 与 cpu-initial.log：首轮 40 项运行，保留两处 smoke fixture 错误。
- cpu-fixture-retest/：修复夹具后两项定向通过。
- cpu-resources/、cpu-foundation-regression/、cpu-plan-regression/：资源及既有 CPU 回归。
- gpu-real/cases.json、*-exit.json、*-cli.log、*-gpu.jsonl、gpu-after.json：冻结输入、退出、日志及显存记录。
- gpu-real/verification.json：32 项真实 job 产物/设备/退出审计；gpu-real/audit-extended.json：214 项只读重建及来源审计。
- 实际冻结记录：smoke/results/infer-orchestrator-20261028-t1qglfpo/gpu-real/records_root/smoke-{zero_shot,multitask}/stage-runs/eval/eval-smoke/；输出在同一来源根目录的 eval_root/ 下，完整绝对地址见 gpu-real/cases.json。本轮结果目录不创建重复 records_root 副本。
- scope-before.json、scope-final.json、verification-summary.json：变更边界审计与总结。

最终范围审计覆盖 209 个非 smoke 文件的内容/权限/链接、HEAD/branch/refs、原工作区 diff/status、暂存内容/index 字节及 smoke 链接；实际核对结果见 scope-final.json 和 verification-summary.json。执行器验证仅覆盖低级 eval job，不包含尚未实现的高级 eval 编排；真实 judge 只在单卡、两种 setting 的各 24 条 smoke 数据上运行，不代表全数据、多卡、其他 setting 或模型质量。未改 production 代码、正式 YAML、环境、输入数据或权重；未 stage、commit、push、merge 或 switch 分支。


## 验证入口与使用方式

CPU/总控 Python 为 /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python，CPU 环境 Python 3.11.6、PyYAML 6.0.3；设置 PYTHONDONTWRITEBYTECODE=1、GIT_OPTIONAL_LOCKS=0、CUDA_VISIBLE_DEVICES=，以 -B 运行。

- scripts/test_eval_executor.py --result-root <全新 smoke/results 子目录>：40 个 CPU eval 用例；可重复指定 --test NAME 定向复测。首轮 cpu-initial 两处夹具错误的定向通过证据保存在 cpu-fixture-retest 中，不把复测当作两个额外独立用例。
- scripts/test_eval_resources.py --result-root <全新子目录>：锁等待取消、容量槽位以及 SIGTERM 下 worker/descendant 清理。
- scripts/run_eval_executor_gpu.py --result-root <本轮 gpu-real> audit：只读复核本轮真实结果；不加载模型。其 setup/run 模式会绑定并执行真实 job，本轮已完成，不应对已有证据再次执行。
- scripts/audit_eval_executor.py --result-root <本轮 gpu-real>：拦截模型导入，用 CPU 完整重算 raw/scored/metrics、输入来源、参数、PID 退出及锁释放。

首轮 CPU 1071.881 秒；只修正两项测试夹具后的定向复测通过。基础回归 44.020 秒、计划回归 15.608 秒、资源用例 22.493 秒。真实 job 每组包含全部六 task，每组 22 MCQ + 1 OPEN + 1 MRA，两组实际均未触发修复重试；修复和解析失败分支由 CPU 合成用例覆盖。
