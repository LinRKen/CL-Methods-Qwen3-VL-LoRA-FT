# COSINE 实验配置与产物基础

本目录实现实验配置、计划、运行记录、产物清单、单个低级 LoRA tune job 的执行器，以及 Multitask、Singletask 的高级 tune 编排。CL/Transfer tune 编排、infer、eval 执行后端仍待后续阶段。YAML 中的参数为待最终核对的初始值。

## 文件组织

- `configs/default.yaml`：共用默认参数。
- `configs/*.yaml`：七种 setting 模板，只填写相对默认配置的差异。
- `lib/config.py`：严格 YAML 读取、递归合并、参数校验和配置快照。
- `lib/records.py`：experiment → 高级 stage run → 低级 job 的记录、依赖和状态接口。
- `lib/artifacts.py`：外部输入及运行产物的来源、精确路径和内容清单。
- `lib/common.py`：ID、时间、JSON、指纹、原子写入和记录锁。
- `lib/tune_executor.py`：校验冻结 tune job、限定设备并执行一次训练，保存日志和运行证据、登记 adapter 与完整 checkpoint。
- `run_tune_job.py`：单 job 命令行入口；不调度同一高级 run 的其他 job。
- `lib/tune_orchestrator.py`：调度 Multitask/Singletask 的冻结 tune jobs，按有效资源分配 GPU 组、管理失败和中断、汇总状态与 adapter 来源。
- `run_tune.py`：高级 tune 入口；从 YAML 新建实验，或执行已有 tune run。只执行训练阶段。
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

plan 只打印 JSON 计划；prepare 创建记录但不执行模型。可使用 --default 指定其他默认 YAML，使用 --set infer.params.batch_size=2 等点号参数临时覆盖。run_tune_job.py 接受准备好的 job 目录和冻结 gpu_groups 的零起始索引。例如：

~~~bash
python scripts/experiments/run_tune_job.py --job-dir /home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/tune/<run-id>/jobs/tune-01-motion --gpu-group-index 0
~~~

未绑定内部依赖时，从成功 producer job 的 adapter manifest 显式传入，参数格式为 `--input <role>=<绝对 manifest 路径>`。例如：

~~~bash
python scripts/experiments/run_tune_job.py \
  --job-dir /home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/tune/<run-id>/jobs/tune-02-hypo \
  --gpu-group-index 0 \
  --input parent-motion=/home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/tune/<run-id>/jobs/tune-01-motion/artifacts/adapter.json
~~~

恢复同一训练任务的中断状态时，给新的 planned job 显式传入失败/取消 attempt 已登记的 `artifacts/checkpoint-<step>.json`，并加 `--resume-training-state`。如果冻结计划还要求父 adapter，也要按其 adapter manifest 一并传入；checkpoint 提供模型权重和训练状态，父 adapter 用于验证 lineage。恢复要求训练 task、有效 job 配置、数据/模型输入指纹、父 adapter payload、DeepSpeed 配置和 GPU world size 与原 attempt 一致。不会搜索 latest checkpoint，也不会覆盖失败 attempt 的输出。终态 job 不能重新启动；低级入口不会重启终态 job，也不自动生成 retry 记录；恢复所需的新 planned job 记录由后续编排层负责创建。

base、processor 和 annotation manifest 按冻结的 input_specs 自动生成并校验；内部 adapter/checkpoint 必须显式指定。每个 job 使用冻结 GPU 分组的零起始索引，设备在启动 torchrun 前解析为获准物理 GPU 的 UUID；物理 GPU 0 会被拒绝，W&B 监控只筛选同一物理设备组。GPU、端口及高级 tune run 的并行槽位使用文件锁协调；端口为 `master_port + gpu_group_index`。job 同步执行并将 stdout/stderr 写入 `logs/train.log`。

每个低级输出目录包含 `checkpoints/`（训练 checkpoint 与 training_result.json）、`adapter/`（最终可复用 LoRA adapter 和 processor）、`logs/`（训练日志及 W&B 文件）及执行收据 `execution.json`。完成后 adapter、完整训练 checkpoint 和 summary manifest 会登记到 job 的 artifacts/；失败或中断时，执行器会尝试保留已完整写入的 checkpoint manifest，作为显式恢复来源。低级执行器只将父高级 run 推进为 running，完成/失败的汇总由高级调度器处理。一次低级命令只运行一个 job。

## Multitask 与 Singletask 高级 tune

`run_tune.py` 将一个 tune 高级 run 的任务交给多个低级 `run_tune_job.py` 进程；高级进程不加载模型。Multitask 只调度 `tune-whole`，使用 whole/train；Singletask 按 motion、hypo、pattern、nav、view、geo 的顺序放入队列，共六个 `tune-<task>`。每个任务从相同原始 base model 新建独立 LoRA、optimizer、scheduler、epoch 和 global step，不继承其他 task 的 adapter。该队列顺序不构成持续学习关系，并行任务的结束顺序可以不同。

从 setting YAML 创建新 experiment、冻结所有已启用阶段的计划及 job，然后只执行 tune。例如（本轮仅提供命令，未执行）：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py \
  --config scripts/experiments/configs/multitask.yaml \
  --experiment-id multitask-example

/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py \
  --config scripts/experiments/configs/singletask.yaml \
  --experiment-id singletask-example
~~~

`--default`、`--set` 的含义与 `pipeline.py prepare` 一致；未指定值沿用默认 YAML。已有 experiment ID 不覆盖，重新实验使用新 ID。入口在创建记录前拒绝 CL、Zero-shot、Transfer 及未启用 tune 的配置。infer/eval 即使已登记，也只保持 planned。

已使用 prepare 建好记录，或先手动完成了其中若干低级 job 时，可指定其高级 tune run 目录：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py \
  --run-dir /home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/tune/<run-id>
~~~

`--run-dir` 不接受 `--config`、`--default`、`--set` 或 `--experiment-id`，只读取冻结记录。接管要求高级 run 仍为 planned/running，所有 job 为未执行的 planned 或已成功的 succeeded；已成功 job 的 adapter 内容和来源通过 manifest 复核后跳过。已有 execution.json/输出的 planned job、running job 或 failed/cancelled job 会被拒绝，需先人工审查。不要同时用低级入口启动同一 run 的任务。高级入口不重置终态、不自动重试、不寻找最新 checkpoint；显式恢复仍使用低级入口绑定 checkpoint 后执行，成功后高级入口可以接管其余 planned jobs。

参数与资源分工：

- 高级 run 的 `tune.run.max_parallel_jobs` 限制并发低级进程数；`tune.run.fail_fast` 决定失败策略。这两个调度参数以高级 run 的冻结值为准，job 局部覆盖不改变整个 run 的策略。
- 每个 job 使用自身合并后的 `tune.params`，以及 `tune.run.python`、`gpu_groups`、`master_addr`、`master_port` 和 W&B 设置；高级 CLI 覆盖继续按原配置优先级生效。
- 调度器从各 job 的有效 `gpu_groups` 选择空闲组，按物理 GPU 集与 `master_addr`/`master_port + group_index` 避免本 run 内的资源冲突；低级执行器继续持有 GPU、端口和阶段槽位锁。job 局部 GPU 分组不同时也按实际设备处理。物理 GPU 0 始终禁止，训练与 W&B 使用同一分组。
- `max_parallel_jobs: 1` 或只有一组 GPU 时依次训练。配置多组独立 GPU 并提高并发上限后可并行；单组可含多张 GPU，由低级执行器使用 torchrun。
- `fail_fast: true` 在发现首个失败后停止派发，并请求其他活动低级进程退出、等待其关闭训练进程及登记已完整写出的 checkpoint；未启动任务标为 cancelled。`false` 继续其他独立任务，最后仍将有失败的高级 run 标为 failed。如果低级进程已经退出而 job 仍为 running，或其执行收据记录清理失败，调度器停止整个 run，保留失败证据供检查残留训练进程，不继续复用该设备。
- SIGINT/SIGTERM 同样停止派发、请求低级清理并等待退出，保留已有成功产物与证据，整体记为 cancelled。高级层不强杀仍在清理或保存 checkpoint 的低级进程。

每次高级调用在 run 记录树内新建 `orchestrations/<invocation-id>/`，保存 `execution.json`、各低级入口的 `<job-id>.log` 和 `summary.json`；低级详细训练日志仍位于原 payload 的 `logs/train.log`。run 根目录的 `tune-summary.json` 保存最新汇总，列出每个 job 的状态、退出码、GPU/端口分配、日志、输出目录及 adapter manifest 路径/指纹/producer/slot。若 adapter 内容或来源复核失败，高级 run 不记录成功，即使该 job 先前已是 succeeded。

所有计划训练任务成功且 adapter 完整后，tune run 才变为 succeeded。本入口将失败或中断记录在 tune run/job 层，不提前结束整个 experiment；experiment 仅在所有已启用阶段均成功时记为 succeeded，默认启用 infer/eval 的实验训练完成后仍为 active。高级 CLI 成功返回 0，失败返回 1，中断返回 130。CL/Transfer 调度、infer/eval 自动运行、自动重试记录和最终 YAML 参数核对继续留待后续阶段。

