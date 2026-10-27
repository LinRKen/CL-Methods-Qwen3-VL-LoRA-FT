# Smoke 测试

本目录用于管理快速验证训练、推理及辅助功能的测试代码、配置、少量固定样本和验证报告。已完成 pipeline 冻结计划修复的 CPU 验证；原有 77 项与新增 26 项共 103 项全部通过，详见 [修复验证报告](reports/PIPELINE_PLAN_FIX_20261027.md)。此前三项失败及原始证据保留在 [第一阶段报告](reports/PIPELINE_FOUNDATION_20261027.md)。

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

第一阶段 CPU 验证入口（在仓库根目录执行，使用全新结果目录）：

```bash
run_dir="smoke/results/pipeline-foundation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
bash smoke/scripts/run_pipeline_foundation.sh "$run_dir"
```

该验证不使用 GPU，不加载真实模型或数据；已有 cases/ 或 tests.json 的结果目录会被拒绝。修复后的版本原有 77 项全部通过。此前失败结果保留在第一阶段报告中。

冻结计划边界验证使用独立的新目录：

```bash
constraint_dir="smoke/results/pipeline-plan-constraints-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$constraint_dir"
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1   /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B   smoke/scripts/test_pipeline_plan_constraints.py --result-dir "$constraint_dir"
```

该入口只运行新增的 26 项测试，复用第一阶段的合成 fixture 和模型导入限制；与原有 77 项合计 103 项。已存在的测试结果不会被覆盖。

后续每项测试应说明：

1. 验证目标及通过条件。
2. 所需环境、依赖和输入；固定样本记录来源、样本 ID 及预期输出。
3. 完整执行命令、运行目录和主要参数。
4. 外部结果保存位置及对应的验证报告。

训练环境路径为 `/home/linruikang/data/envs/Qwen3VL-LoRA-FT`；需要其他环境时，在相应测试说明和报告中记录实际使用的环境。创建本目录不代表环境或功能验证通过。

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

## 低级 LoRA 训练执行器验证

2026-10-27 验证通过：CPU 28 项通过；物理 GPU 1 上新建 LoRA、跨任务 adapter 继承和完整 checkpoint 恢复均成功。恢复后的 adapter 与未中断训练逐张量一致；范围审计确认 smoke 外的仓库状态未变。详情、命令和限制见 [执行器验证报告](reports/TUNE_EXECUTOR_20261027.md)。

验证代码为 `scripts/test_tune_executor.py` 和 `scripts/run_tune_executor_gpu.py`。CPU 入口屏蔽 GPU 并使用合成产物；GPU 入口固定物理 GPU 1，会实际训练三个短 job。两者都要求预先创建全新的 smoke/results 子目录。GPU 入口的 `--verify-only` 仅审计已有训练产物，不重新训练。
