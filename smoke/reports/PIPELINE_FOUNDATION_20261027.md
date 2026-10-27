# Pipeline 第一阶段 CPU 验证（2026-10-27）

## 结论

验收未全部通过：77 项测试中 74 项通过、3 项失败、0 项错误，最终测试耗时 42.572 秒，退出码 1。三个失败均为实验计划与运行记录的一致性缺陷，保留原断言与失败证据，本轮未修改生产代码。

## 测试对象与环境

- 服务器：885；仓库：/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT。
- 分支：test/pipeline-automation。
- HEAD：043f3d2d8626ac54ef7311c32957b248929222d8。
- 被测功能提交：096418ca50ee1f9e1f2fec190b1b25f7003cc58d。
- Python：3.11.6；PyYAML：6.0.3。
- Python 路径：/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python。
- 纯 CPU；CUDA_VISIBLE_DEVICES 为空，禁止模型/GPU 包导入，最终 forbidden_imports=[]。
- 只创建合成配置、微小假权重文件与记录，不读取真实模型或 COSINE 数据，不执行训练、推理、评测或 GPU 查询，不安装依赖。

## 已通过范围

- 七种 setting 模板及自定义 CL 顺序，任务数、依赖与父 adapter 来源正确。
- CL 两模板各 36 个评测对；zero-shot、multitask、singletask 各 6 对；transfer 两模板各 30 对。transfer tune 计划含第一轮 6 个和第二轮 30 个 job。
- infer 按 adapter 分组、每组一个 job；eval 将全部评测对纳入一个 judge job。这里只验证计划，不代表真实 vLLM 加载次数已实测。
- 默认配置、setting、job、高级与局部 CLI 的覆盖优先级；false/0 保留、列表替换、深拷贝、来源指纹与快照冻结。
- 42 类非法配置拒绝，包括 tune/infer/eval 的物理 GPU 0、重叠卡组及 TP 不匹配。
- 正式 CLI plan/prepare、七模板 prepare 完整性，未启动模型工作负载。
- 合成输入绑定、来源/任务约束、依赖与状态、LoRA 权重继承和同任务/同配置 checkpoint resume。
- infer 各 task 预测齐全约束、合成 prediction 到 metrics 的登记流程。
- 各类产物 manifest、完整 SHA256、软链接身份及篡改检测。

## 失败项与影响

三个缺陷均在 scripts/experiments/lib/records.py 的记录 API 边界复现：

1. test_stage_plan_rejects_unplanned_job：冻结六-job stage plan 后，create_job 仍接受 not-in-plan；预期拒绝，实际未抛 ValueError。
2. test_stage_plan_rejects_changed_task：将已计划的 tune-01-motion 的 task 改成 geo，create_job 仍接受；未强制登记字段与冻结计划一致。
3. test_stage_success_requires_all_planned_jobs：六-job 计划只注册并完成首个 job，update_status(run, 'succeeded') 仍成功；仅检查已登记子任务，没有确认计划内 job 全部存在且完成。

影响：计划外或任务错误的 job 可以被登记，未完成全部计划的 stage 可能被标记成功。标准 prepare_records 完整建好全部 job 的路径通过，但记录 API 仍缺少计划绑定与完整性检查。建议在 feat 分支修复后，回 test 分支运行同一组回归。

## 复现

在仓库根目录执行；目录必须预先创建且不能含旧 cases/ 或 tests.json：

```bash
run_dir="smoke/results/pipeline-foundation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
bash smoke/scripts/run_pipeline_foundation.sh "$run_dir"
```

当前版本预期退出码为 1，报告上述三个失败。不要覆盖本轮证据或为得到通过结果删除断言。

## 证据

仓库入口：smoke/results/pipeline-foundation-20261027-134504/。

实际位置：/mnt/hdd2/linruikang/smoke/CL/CL-Methods/Qwen3-VL/LoRA-FT/results/pipeline-foundation-20261027-134504。

- baseline.json：测试前 Git 状态、非 smoke 文件哈希、权限与链接。
- tests.json、tests.log：首轮 74 项通过的原始证据，保留不覆盖。
- acceptance-final/tests.json、tests.log：增加三项计划一致性断言后的最终 77 项结果。
- acceptance-final/cases/：各项合成配置、产物、记录和 CLI 日志。
- scope-audit.json：测试后首次范围核验。
- scope-audit-final.json：报告与 README 收尾后的范围核验。

两-job 继承生命周期测试明确作为无 stage plan 的独立 API 测试；六-job 计划完整性由三个独立失败测试覆盖，没有通过弱化预期隐藏缺陷。

## 修改范围与限制

本轮仅新增 smoke/scripts/audit_pipeline_scope.py、run_pipeline_foundation.sh、test_pipeline_foundation.py 和本报告，并更新 smoke/README.md。完整运行产物通过 smoke/results 外置，不纳入 Git。

范围核验检查 HEAD、分支、refs、原暂存区/index、非 smoke diff/status、197 个非 smoke 文件的内容/权限/链接及 smoke 软链接保持，另检查 git diff --check。未修复生产代码，未暂存、commit、push、merge 或切换分支。

本阶段验证配置、计划、记录与产物基础，不包含真实模型的训练、推理、评测执行或真实 LoRA 权重加载。
