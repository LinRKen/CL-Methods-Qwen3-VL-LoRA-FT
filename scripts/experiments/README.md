# COSINE 实验配置与产物基础

本目录实现实验配置、计划、运行记录和产物清单。当前阶段不包含 tune、infer、eval 执行后端或资源调度。YAML 中的参数为待最终核对的初始值。

## 文件组织

- `configs/default.yaml`：共用默认参数。
- `configs/*.yaml`：七种 setting 模板，只填写相对默认配置的差异。
- `lib/config.py`：严格 YAML 读取、递归合并、参数校验和配置快照。
- `lib/records.py`：experiment → 高级 stage run → 低级 job 的记录、依赖和状态接口。
- `lib/artifacts.py`：外部输入及运行产物的来源、精确路径和内容清单。
- `lib/common.py`：ID、时间、JSON、指纹、原子写入和记录锁。
- `pipeline.py`：根据 setting 生成计划，或创建计划及配置记录。
- `requirements.txt`：配置层的 PyYAML 依赖，不替代各模型环境的依赖。

## 参数继承

优先级从低到高：默认 YAML → setting YAML → job 覆盖 → CLI 覆盖。CLI 内部按高级 run 通用覆盖 → 高级 run 的局部 job 覆盖 → 低级 job 覆盖的顺序生效，且均高于 job YAML。生成计划时会先解析各 job 合并后的有效配置，再创建运行记录。

映射递归合并；列表整体替换；false 和 0 是实际值，不代表缺省。省略字段表示继承；null 不受支持。重复 YAML key、未知字段及无效取值会被拒绝。路径使用绝对路径，不执行环境变量或 shell 展开。

配置分为 experiment、tune、infer、eval。各 stage 的 params 保存模型与实验参数，run 保存 Python 路径、GPU 分组、并行上限等运行参数，jobs 按生成的 job ID 指定局部 params/run 覆盖。job 不得重写 experiment 的模型、数据和产物根目录，也不得通过局部参数替换数据或权重继承关系。

局部覆盖示例（可放在对应 setting YAML）：

~~~yaml
infer:
  jobs:
    infer-motion:
      params:
        batch_size: 2
~~~

每个 experiment、stage run 和 job 保存有效配置快照及配置指纹；experiment 快照记录默认/setting 文件的绝对路径与 SHA-256。恢复读取已有快照，不重新加载后来修改的默认 YAML。

run.gpu_groups 只能显式使用物理 GPU 1–3，各组不得重叠；物理 GPU 0 禁止使用。该配置约束不替代后续执行器对实际设备身份的检查。

## 运行层级与计划

一个 setting 实例对应一个 experiment；每个已启用 stage 创建一个高级 run；高级 run 包含多个低级 job。

- tune：每次训练对应一个 job。CL 和 transfer 的后续训练显式引用父 job 的 adapter。
- infer：每个 adapter 对应一个 job，一次加载模型完成其全部测试 task；zero-shot 使用一个 base job。每个测试 task 单独登记 prediction manifest。
- eval：每个 setting 使用一个 judge job，计划包含所有 prediction/task 对；judge 一次加载处理全部预测。

| Setting 模板 | Tune jobs | Infer jobs | 预测/评测对 |
| --- | ---: | ---: | ---: |
| cl_mhpnvg / cl_custom | 6 | 6 | 36 |
| zero_shot | 0 | 1 | 6 |
| multitask | 1（whole） | 1 | 6 |
| singletask | 6 | 6 | 6 |
| transfer_round1 | 36（第一轮 6 + 第二轮 30） | 30 | 30 |
| transfer_round2 | 36（第一轮 6 + 第二轮 30） | 30 | 30 |

CL 两个模板均使用 setting: cl，通过 task_order 控制训练顺序；cl_mhpnvg 默认为 motion、hypo、pattern、nav、view、geo。transfer_round1 测第二轮 adapter 在第一轮 task 上的表现；transfer_round2 测其在第二轮 task 上的表现。第一轮 adapter 也需保留，供第二轮分支使用。

depends_on 仅表示同一高级 run 内的 job 执行先后；requires 描述必须绑定的 stage/job/kind/slot/task 产物。input_specs 则声明外部 base/judge model、processor 和每项 COSINE train/test 标注文件的角色、精确路径、来源及 task/split。job 启动前必须把这些 manifest 按角色显式绑定。跨 stage 通过产物引用衔接，不搜索最新 checkpoint。绑定内部产物时检查 experiment/stage/job 和 slot/task；manifest 本身保留精确 producer（experiment/stage/run/job）。一个 experiment 每个 stage 只建一个高级 run，重复实验用新的 experiment ID，避免同名 job 指向不同 stage run。

跨 task 采用继续训练同一个已有 LoRA adapter 的方式，只继承参数，重置 optimizer、scheduler、epoch 和 global step。仅同一 task 的中断恢复可使用完整 checkpoint，并要求训练 task 及有效配置指纹一致；恢复时父 adapter 的清单仅保留来源关系，权重和训练状态都取自 checkpoint。

## 记录与产物

记录写入 experiment.records_root/<experiment_id>/，主要结构：

~~~text
<experiment_id>/
  experiment.json
  config.snapshot.json
  plan.json
  stage-runs/<stage>/<run_id>/
    run.json
    config.snapshot.json
    plan.json
    jobs/<job_id>/
      job.json
      config.snapshot.json
      artifacts/<role>.json
~~~

实际 payload 按 <stage 对应根目录>/<experiment_id>/<run_id>/<job_id>/ 隔离，分别使用 checkpoints_root、infer_root 和 eval_root。prepare 只建立记录，不创建模型 payload 或复制权重。

manifest 保存精确文件/目录、文件大小、完整 SHA-256 和来源：内部产物指定 producer，外部输入指定 external_source，二者必须且只能有一个。目录按全部文件记录内容；内部目录软链接不支持。相同 payload 的 manifest 可以复用，不需要复制模型。

主产物分别为 tune 的 adapter、infer 的逐 task prediction、eval 的 metrics。也支持 tune checkpoint、judge raw/scored 结果和 summary 等辅助产物。job 只有登记完整主产物后才能记录成功；父 run/experiment 只有所有子项成功后才能记录成功。状态接口仅记录元数据，不启动、停止或调度进程。

## 入口示例

以下命令用于后续使用，本阶段没有执行：

~~~bash
cd /home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT
python scripts/experiments/pipeline.py plan --config scripts/experiments/configs/cl_mhpnvg.yaml
python scripts/experiments/pipeline.py prepare --config scripts/experiments/configs/cl_mhpnvg.yaml --experiment-id cl-mhpnvg-example
~~~

plan 只打印 JSON 计划；prepare 创建记录但不执行模型。可使用 --default 指定其他默认 YAML，使用 --set infer.params.batch_size=2 等点号参数临时覆盖。真实运行入口、权重加载与续训、模型执行及恢复调度在后续阶段接入。

