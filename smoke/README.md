# Smoke 测试

本目录用于管理快速验证训练、推理及辅助功能的测试代码、配置、少量固定样本和验证报告。已新增 COSINE Qwen 4B 评测测试；流程验证通过，语义判分存在未通过项，详见下方报告。

## 目录结构

```text
smoke/
├── README.md
├── .gitignore
├── scripts/       # smoke 测试代码
├── configs/       # 测试配置
├── fixtures/      # 少量固定样本及预期输出
├── reports/       # 需要纳入 Git 的验证报告
├── data           # 指向外部 data 目录的软链接
└── results        # 指向外部 results 目录的软链接
```

各子目录中的 `.gitkeep` 用于让 Git 保留空目录；添加实际文件后可按需移除。

## 外部数据与结果

外部存储根目录：

```text
/home/linruikang/data/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT
```

885 已建立以下两个软链接，用于访问外部目录：

| 仓库内入口 | 指向的外部目录 |
| --- | --- |
| `smoke/data` | `/home/linruikang/data/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/data` |
| `smoke/results` | `/home/linruikang/data/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/results` |

软链接指向既有外部目录，创建链接不修改目标目录中的内容。链接由本目录的 `.gitignore` 忽略；其他机器 clone 后，需自行准备外部目录及软链接。

## Git 管理范围

- 纳入 Git：本说明、`.gitignore`、测试代码、配置、少量固定样本及预期输出、选定的验证报告。
- 不纳入 Git：`data` 和 `results` 入口及其外部内容，包括大体积数据、完整日志、批量预测结果、临时产物和权重。
- `smoke/.gitignore` 使用 `/data` 和 `/results`，不加末尾斜杠，以便忽略软链接本身。Git 不会沿软链接跟踪外部文件。
- 报告只引用外部结果位置，不把完整运行产物复制进仓库；提交前确认不含密钥、令牌等敏感信息。

## 添加与运行测试

现有脚本与复现命令见下节。每项测试应说明：

1. 验证目标及通过条件。
2. 所需环境、依赖和输入；固定样本记录来源、样本 ID 及预期输出。
3. 完整执行命令、运行目录和主要参数。
4. 外部结果保存位置及对应的验证报告。

训练环境路径为 `/home/linruikang/data/envs/Qwen3VL-LoRA-FT`；需要其他环境时，在相应测试说明和报告中记录实际使用的环境。创建本目录不代表环境或功能验证通过。

## COSINE Qwen 4B 评测验证

2026-10-26 已在 test/eval-qwen 验证：CPU 30 项通过，motion 691 条及断点续跑通过；固定语义用例仅 22/30 完整符合预期，存在误判，不能视作判分可靠性验收通过。详情见 [验证报告](reports/EVAL_QWEN_4B_20261026.md)。

| 文件 | 用途 |
| --- | --- |
| scripts/test_eval_qwen.py | CPU 行为与异常分支检查 |
| fixtures/eval_qwen_cases.json | 30 条运行前固定的人工用例和预期 |
| configs/eval_qwen_4b.json | 4B、物理 GPU 1 与已有 motion 输入配置 |
| scripts/run_eval_qwen_fixtures.py | 使用生产函数验证人工用例 |
| scripts/interrupt_eval_qwen.py | 内存包装生产函数，在 64 条完整写入后主动中断 |
| scripts/run_eval_qwen_integration.py | 中断、正式入口续跑、完整结果无 GPU 重建和产物检查 |
| scripts/eval_qwen_audit/sitecustomize.py | 记录 worker 最早期的设备限制，只在测试配置下启用 |

在 885 的仓库根目录复现（会重新启动 GPU 评测；结果写新的外部目录）：

```bash
cd /home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT
EVAL_PY=/home/linruikang/data/envs/Qwen3VL-vLLM-Verify-20261025/bin/python
RUN_ROOT=$(mktemp -d /home/linruikang/data/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/results/eval-qwen-4b-XXXXXXXX)
export CUDA_VISIBLE_DEVICES=GPU-63a25b73-5c01-919f-188a-29c116904574
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/qwen-vl-finetune"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='' "$EVAL_PY" smoke/scripts/test_eval_qwen.py --report "$RUN_ROOT/cpu.json"
"$EVAL_PY" smoke/scripts/run_eval_qwen_fixtures.py --output-dir "$RUN_ROOT/fixtures"
"$EVAL_PY" smoke/scripts/run_eval_qwen_integration.py --output-dir "$RUN_ROOT/motion"
```

三个命令分别运行；固定用例脚本退出 2 表示实际语义结果未全部符合预期，不应据此修改预期用例。集成脚本内部的主动中断退出 75 为预期，最终集成脚本成功退出 0。每个输出子目录必须不存在，避免覆盖既有证据。需已有配置指定的本地权重、motion 推理结果和 COSINE 测试集，不下载依赖或数据。CPU 测试无需 GPU；GPU 脚本显式固定为物理 GPU 1。

## COSINE Qwen 8B 重评

配置位于 configs/eval_qwen_8b.json，沿用同一组人工用例及原 motion checkpoint-543 预测，只更换 judge 权重。两个 runner 均支持 --config，默认仍读取 4B 配置。复现时按上面的环境设置，建立新的 RUN_ROOT，然后顺序运行：

~~~bash
"$EVAL_PY" smoke/scripts/run_eval_qwen_fixtures.py --config smoke/configs/eval_qwen_8b.json --output-dir "$RUN_ROOT/fixtures"
"$EVAL_PY" smoke/scripts/run_eval_qwen_integration.py --config smoke/configs/eval_qwen_8b.json --output-dir "$RUN_ROOT/motion"
~~~

本次 8B 用例完整符合预期 26/30、最终分数符合预期 30/30，仍有状态及数值抽取偏差。motion 为 159/691（23.01%），invalid 从 4B 的 527 降到 5，逐条最终分数不变。详情见 reports/EVAL_QWEN_8B_20261026.md。此结果不能解释为语义验收全部通过。

## 验证报告

在 `reports/` 中按测试主题保存简洁报告，至少记录：

- 测试时间、目标和实际使用的代码提交；若有未提交改动，说明改动范围。
- 环境和关键依赖版本、命令、配置、样本范围。
- 预期行为、实际结果、退出码及通过或失败结论。
- 外部结果目录和关键证据文件位置。
- 已验证范围、失败项和仍未验证的限制。

不得将计划中的测试或尚未返回的运行写成已通过。

## GPU 使用约束

未经用户明确授权，禁止使用 885 的物理 GPU 0；适用于训练、推理、评测及 GPU 诊断或测试。

所有 GPU 进程启动前，必须显式限制到物理 GPU 1–3 中所需的子集，按物理编号及 UUID 核实设备，并确认启动器和子进程继承限制。禁止默认使用全部 GPU 或回退到物理 GPU 0。进程内逻辑 `cuda:0` 可以对应获准的物理 GPU，不能仅凭逻辑编号判断。
