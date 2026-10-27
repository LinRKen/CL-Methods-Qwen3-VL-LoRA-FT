# Pipeline 冻结计划修复验证（2026-10-27）

## 结论

验证通过：原有 77 项 CPU 回归测试全部通过，针对本次修复新增的 26 项边界测试全部通过，共 103 项通过、0 项失败、0 项错误。

## 测试对象与环境

- 服务器：885；仓库：/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT。
- 分支：test/pipeline-automation。
- HEAD：dca43300b156abff858e04292db27b8c25267355。
- 被测合并提交：dca43300b156abff858e04292db27b8c25267355；修复提交：733691c9d367ee14367077f96cc2e22ecc37941f。
- Python：/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python，Python 3.11.6，PyYAML 6.0.3。
- 纯 CPU，CUDA_VISIBLE_DEVICES 为空；测试禁止导入 torch、vllm、transformers、deepspeed、wandb、peft 和 qwen_vl_utils，实际 forbidden_imports 为空。

## 执行结果

| 测试 | 项数 | 失败 | 错误 | 耗时 | 退出码 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原有 CPU 回归 | 77 | 0 | 0 | 43.059 秒 | 0 |
| 冻结计划边界回归 | 26 | 0 | 0 | 15.725 秒 | 0 |

原有 test_stage_plan_rejects_unplanned_job、test_stage_plan_rejects_changed_task 和 test_stage_success_requires_all_planned_jobs 保留原断言，均已通过。原始失败报告与结果目录不变。

## 验证范围

- 重跑第一阶段原有 77 项测试，覆盖七种 setting 计划、CLI/YAML 配置覆盖、记录快照与指纹、输入绑定、产物登记、继承与状态生命周期。
- 新增 26 项冻结计划边界测试，覆盖计划外 job、重复 job、缺失/额外/重复登记、job 路径与 ID 不一致、七个计划字段变更、计划字段缺省值、配置覆盖、运行时输入、完整六 job CL 计划以及计划篡改。
- 检查失败操作不创建目录、不修改 run.json 或 plan.json，stage 成功状态只在计划完整且所有 job 成功后写入。
- 范围审计通过，非 smoke 文件无变化。

## 复现

在仓库根目录执行；两个结果目录均需为新目录：

```bash
run_dir="smoke/results/pipeline-plan-fix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
bash smoke/scripts/run_pipeline_foundation.sh "$run_dir"
mkdir "$run_dir/constraints"
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1   /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B   smoke/scripts/test_pipeline_plan_constraints.py --result-dir "$run_dir/constraints"
```

## 结果证据

- 原有回归：smoke/results/pipeline-plan-fix-20261027-FEEluTy7/tests.json。
- 新增边界回归：smoke/results/pipeline-plan-fix-20261027-FEEluTy7/constraints/tests.json。
- 范围审计：smoke/results/pipeline-plan-fix-20261027-FEEluTy7/scope-audit-complete.json。
- 结果目录实际位于：/mnt/hdd2/linruikang/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/results/pipeline-plan-fix-20261027-FEEluTy7。

## 修改范围

仅新增 smoke/scripts/test_pipeline_plan_constraints.py、本报告，并更新 smoke/README.md；生产代码没有新增修改。外部结果目录由 smoke/results 软链接访问，未纳入 Git。未使用 GPU，未执行训练、推理或评测，未 commit、push 或 merge。

## 限制

本轮只验证实验配置、冻结计划、记录 API 和产物生命周期，不执行真实模型 workload，也不验证真实数据、LoRA 权重或 vLLM。
