# 高级编排预检失败日志修复验证（2026-10-30）

本轮在 885 的 `test/pipeline-automation` 验证修复提交
`4b744c011e94df8e3d4408666b5e5e0673461a18`。合并后的 HEAD 为
`34aed75add126ac427aedba431c590575d6d8389`。

## 验证结果

七组 CPU 测试共 253 个独立用例，全部通过；失败和错误均为 0。测试使用合成记录，没有加载模型。

| 套件 | 用例数 | 结果 |
| --- | ---: | --- |
| 预检日志定向回归 | 36 | 全部通过 |
| 初始化期间有效快照替换 | 3 | 全部通过 |
| 高级 infer 编排回归 | 40 | 全部通过 |
| 高级 tune 编排回归 | 27 | 全部通过 |
| 高级 eval 编排回归 | 44 | 全部通过 |
| 配置、记录和产物基础回归 | 77 | 全部通过 |
| 冻结计划约束回归 | 26 | 全部通过 |

完整套件结果和禁止模型导入检查见 `verification-summary.json`。分项证据位于本轮结果根：

`smoke/results/pipeline-logging-fix-20261030-094622-3670397/`

定向 invocation 的日志产物和记录 PID 另见 `final-artifact-audit.json`。

## 本次修复的直接覆盖

- 对 tune、infer、eval 分别注入计划指纹篡改、计划缺失、JSON 损坏和重复 job ID。校验失败前不启动低级 worker，失败后生成 invocation、failure 和 orchestrator 三类日志。
- 对阶段不匹配和初始化期间 run 元数据变化分别检查错误记录、异常重抛和 callback 未执行。
- 三项有效快照替换测试只改变来源信息并重建合法快照指纹，配置指纹和 run.json 保持不变；严格 loader 仍能加载，但日志包装器检测到初始化期间快照被替换并拒绝执行。
- 检查成功返回值、stdout/stderr tee、执行异常、KeyboardInterrupt 对应的 cancelled 状态。
- 检查 metadata loader 可读取坏计划，同时原严格 loader 继续拒绝，不放宽冻结计划校验。
- 三个真实高级 CLI 子进程分别以退出码 1 拒绝坏计划，并保存三类日志、traceback；测试确认冻结记录未改写。
- 边界检查：配置快照自身指纹失效时，在信任 logs_root 前拒绝，不创建 invocation。这不属于“元数据可读后计划预检失败”的日志保证范围。

33 次 invocation 的 96 份日志产物独立重读通过：failed 27、cancelled 3、succeeded 3；三个 CLI 退出码均为 1，五个记录 PID 均已退出，定向测试进程均已结束。

## 测试代码与范围

新增 `smoke/scripts/test_pipeline_logging_fix.py`，可用 `--test <方法名>` 将指定用例分别应用到 tune、infer、eval 三阶段。更新基础 smoke 测试夹具，将 `logs_root` 也限定在每个用例目录内，避免触及正式日志位置。

仅修改 smoke README、定向测试脚本、基础夹具和本报告；保留已有训练脚本改动及未跟踪缓存/W&B 文件。没有改生产代码、正式/default YAML、环境、原数据或权重，也没有暂存、commit、push、merge 或 switch。

范围基线 `scope-before-additional-tests.json` 在 infer 回归启动后、追加的定向测试和其他回归启动前捕获，覆盖 213 个非 smoke 文件。最终审计 `scope-final-report-update.json` 的 10 项检查全部通过：基线中的非 smoke 文件哈希/权限/链接、HEAD、分支、refs、index、diff、暂存区及原非 smoke 状态均保持，`git diff --check` 通过。因为基线不是在 infer 启动之前捕获，不把它表述成覆盖整个测试期间的完整前后快照。

## 执行环境与限制

所有测试使用 `CUDA_VISIBLE_DEVICES=`、`PYTHONDONTWRITEBYTECODE=1`、`GIT_OPTIONAL_LOCKS=0` 和 `python -B`。解释器为 `/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python`。定向、基础和冻结计划套件均未导入 torch、vLLM、transformers、DeepSpeed、W&B、PEFT 或 Qwen video utilities。

编排回归替换低级 worker，日志里的 GPU 分组仅为合成资源配置。本轮没有 GPU 查询、计算、真实训练、推理或 judge 模型加载；GPU 0 未使用。结论限于高级编排日志修复及 CPU 回归，不代表模型效果、全量数据或真实多卡验证。

首次测试启动包装器的 `printf` 分隔符产生了格式告警，但测试进程本身的 infer/tune 退出码均为 0，且各自 JSON 汇总通过。一次只读产物审计误用系统 Python，因其 pathlib 版本缺少 `is_relative_to` 而未生成结果；改用项目环境后审计通过。这些是 smoke 辅助命令问题，未导致生产代码变更或重复运行已成功套件。

定向验证复现方式（使用全新的 smoke/results 子目录）：

```bash
cd /home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
  /home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B \
  smoke/scripts/test_pipeline_logging_fix.py \
  --result-root smoke/results/<new-result-directory>
```
