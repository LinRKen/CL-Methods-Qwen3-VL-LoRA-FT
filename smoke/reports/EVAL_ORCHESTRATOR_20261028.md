# 第九步：实验评测编排与结果汇总验证

日期：2026-10-28，Asia/Shanghai。分支：`test/pipeline-automation`。本轮验证完成，未提交。

## 代码与证据

- 仓库：885 的 `/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT`。
- 验证 HEAD：`464b1dcd33f6c34f3cad911043657ed301825d8a`；第九步功能提交：`83659662e06177e503b30b2829cbc86dba79fb43`。
- 本轮结果根目录：`smoke/results/eval-orchestrator-20261028-pYw7QiHv/`。下文证据路径均相对于此目录。
- 新增 smoke 验证脚本：`scripts/test_eval_orchestrator.py`、`scripts/run_eval_orchestrator_adoption.py`、`scripts/audit_eval_orchestrator.py`。另更新 smoke README 与本报告。

## 结论

本轮覆盖范围内未发现第九步生产代码问题。CPU 共覆盖 150 个独立用例；第九步首轮 44 项中 37 项通过、7 项因同一 smoke 夹具问题失败，修正夹具后这 7 项定向复测全部通过，不重复计入用例数。基础、冻结计划和资源回归共 106 项全部通过。

生产高级 CLI 还接管了两份既有成功的真实 8B judge 产物，完成输入和产物复核、状态收尾及结果表导出；128 项接管审计通过。独立重建 21 次成功高级 invocation 的结果，共 1,292 项输出与进程审计通过，441 个记录 PID 均已不存在。审计项包含重复输出核对，不与 CPU 用例数合并计数。

本轮没有重新加载真实 judge 或启动 GPU 计算。物理 GPU 1 有其他使用者任务，因此真实链路采用既有产物接管；其余 setting 的调度、失败与中断路径使用 CPU 夹具。不能据此宣称新 judge 初始化、全数据、多卡或模型语义质量已通过验证。

## CPU 验证

| 套件 | 独立用例 | 结果 | 主要证据 |
| --- | ---: | --- | --- |
| 第九步高级 eval | 44 | 首轮 37 通过；7 项修正夹具后定向复测通过 | `cpu/summary.json`、`cpu.log` |
| 通用场景定向复测 | 5，已计入上面的 44 | 全过，退出码 0，223.06 秒 | `cpu-retry-common/summary.json` |
| Transfer-round1 定向复测 | 1，已计入上面的 44 | 通过，退出码 0，294.08 秒 | `cpu-retry-round1/summary.json` |
| Transfer-round2 定向复测 | 1，已计入上面的 44 | 通过，退出码 0，292.83 秒 | `cpu-retry-round2/summary.json` |
| 基础回归 | 77 | 全过，退出码 0 | `foundation/tests.json` |
| 冻结计划回归 | 26 | 全过，退出码 0 | `plan-constraints/tests.json` |
| 资源回归 | 3 | 全过，退出码 0 | `resources/summary.json` |
| 合计 | 150 | 修正 smoke 后全部通过 | `verification-summary.json` |

覆盖内容：

- Zero-shot、Multitask、Singletask、CL 与两种 Transfer 的完整计划绑定和结果格子，分别为 6、6、6、36、30、30 对。CL 覆盖自定义顺序；Transfer-round1 测 source task，round2 测第二轮 task。
- 一个高级 eval run 只调度一个低级 judge job；CPU 假 engine 的加载和关闭各一次；成功低级 job 经审计后接管，不重新 spawn。
- 长表 JSON/CSV 与矩阵一致，整体以 `score_sum / total` 加权，包含 MRA 小数分、invalid、解析失败和修复；实际 0 分与未测 `null`/CSV 空串分离。
- 冻结配置在原 YAML 删除后继续有效；预测、GT、producer、登记路径、任务计划或顺序篡改会被拒绝。
- 已运行或终态 job、不完整 receipt/request/output 保护，失败不发布成功表，部分导出失败清理，高级锁和启动异常。
- CLI 成功/失败/中断返回码 0/1/130；SIGTERM 向低级及 worker 传递并等待退出；GPU 0、非法 GPU 组及 TP 不匹配拒绝。

首轮失败均来自 `standard()` 的 `launch_cuda` 断言：旧 fixture 导入会清空 `CUDA_VISIBLE_DEVICES`，新夹具在导入后才记录环境，错误地记成空串。生产启动记录与 receipt 实际为 `1`。修复是在旧 fixture 导入之前保存启动时的环境值；只改 smoke，没有修改生产逻辑或放宽预期。原始失败日志与初始/修正脚本均保存在 `fixture-revisions/`。

## 真实 8B 产物接管

使用第七步 experiment 下第八步已完成的六 task 评分；每 task 四条，共两组 48 条。高级入口实际调用 `scripts/experiments/run_eval.py --run-dir <冻结 eval run 的绝对路径>`，没有重新评分。具体命令与 run 路径见 `real-adoption/cases.json`。

| Setting | 总分 / 样本 | combined_flat_rawacc | invalid | 解析失败 | CLI 退出情况 |
| --- | --- | --- | ---: | ---: | --- |
| Zero-shot | 5.9 / 24 | 24.58% | 14 | 0 | 进程退出、登记成功；退出码无法恢复 |
| Multitask | 7.9 / 24 | 32.92% | 12 | 0 | 退出码 0 |

每个 case 的 64 项审计全部通过，包括成功低级 job 接管、没有新低级子进程、历史模型加载数为 1、完整六对、raw/scored/metrics、独立重算、长表/矩阵、导出 SHA、来源及 lineage。每 case 88 个低级/infer/输入文件的内容哈希与权限保持不变；两份高级 run 和 experiment 均最终登记成功。

Zero-shot 首次 smoke wrapper 的 `wait(timeout=180)` 太短，wrapper 超时退出而生产 CLI 继续运行。未 kill 或重复启动 CLI；仅修 smoke runner 去除短超时并加入 `--resume`，等待原 PID 结束、复核其最终成功状态和全部输出。由于已失去原 wait 父进程，不能取回退出码，证据中保留 `returncode: null`，不能描述为该 CLI 已收集到退出码 0。原失败日志保留在 `real-adoption.log`，续接日志为 `real-adoption-resume.log`。

这些分数沿用既有 smoke 结果，非本轮新增模型效果证据。原 infer 生成上限为 32 tokens，存在输出未答完/截断；分数只用于验证汇总一致性。

## 独立输出与进程审计

`output-process-audit.json` 保存 1,292 项核对：21 次成功高级 invocation（包括初轮已导出成功但夹具末尾断言失败的案例、定向复测及两个真实接管）全部复核导出哈希、逐条分数、invalid 分母、任务格子、CL/Transfer lineage、JSON/CSV 一致性与加权整体分数。441 个从执行 receipt、worker 记录和 runner 记录收集的 PID 全部为 `/proc` 不存在；中断测试的三个层级 PID 另有原始 `pid-audit.json`。

最终范围审计 10 项全部通过：211 个非 smoke 文件的哈希、权限及链接保持，HEAD/branch/refs、非 smoke diff/status、暂存内容、index 二进制哈希及 smoke 链接均与原始基线一致，`git diff --check` 通过。远端仅一个 README、三个验证脚本和本报告变化。逐项证据为 `scope-final.json`，原始基线为 `scope-before.json`，最终汇总为 `verification-summary.json`。

## 命令与复现边界

以下命令均在仓库根目录使用 `/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B`，并设置 `CUDA_VISIBLE_DEVICES=`、`PYTHONDONTWRITEBYTECODE=1`、`GIT_OPTIONAL_LOCKS=0`。CPU 用例的模型和 GPU 身份是显式 fake，不调用真实 CUDA。

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
  /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B \
  smoke/scripts/test_eval_orchestrator.py --result-root <尚不存在的 smoke/results 子目录>

CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
  /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B \
  smoke/scripts/audit_eval_orchestrator.py \
  --result-root smoke/results/eval-orchestrator-20261028-pYw7QiHv \
  --report <该结果目录下尚不存在的 JSON 文件>
```

定向复测的准确命令、PID 和退出码见 `retry-launches.json`，基础回归命令见 `regression-launches.json`。`run_eval_orchestrator_adoption.py --result-root <路径>` 仅用于本报告记录的既有 smoke 计划；`--resume` 用于续接已经开始的接管，不重复运行已成功的高级 run。实际接管已完成，不应再次启动。

## 范围与未覆盖事项

远端改动仅限 smoke 的验证脚本、README、报告和外部结果；没有修改生产代码、正式/default YAML、环境、原始数据或权重，也未暂存、commit、push、merge 或切换分支。禁止使用物理 GPU 0 的规则保持，本轮没有新 GPU 计算，没有干预其他训练任务。

验证范围为高级编排、严格来源绑定、产物接管、计分汇总和故障清理。未覆盖重新初始化真实 vLLM judge 的第九步完整 GPU 路径、全量数据、真实多卡或模型语义质量；第八步已有真实 judge 加载测试不冒充为本轮新增验证。最终 default/setting YAML 参数核对留待用户安排，本轮不自动执行或提交。
