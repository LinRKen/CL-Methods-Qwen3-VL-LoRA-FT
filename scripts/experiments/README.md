# COSINE 实验配置与产物基础

本目录实现实验配置、计划、运行记录、产物清单，低级 LoRA tune、Adapter/base infer 和批量 Qwen 柔性 eval 执行器，以及高级 tune、infer 和 eval 编排与结果汇总。YAML 中的参数为待最终核对的初始值。

## 文件组织

- `configs/default.yaml`：共用默认参数。
- `configs/*.yaml`：七种 setting 模板，只填写相对默认配置的差异。
- `lib/config.py`：严格 YAML 读取、递归合并、参数校验和配置快照。
- `lib/records.py`：experiment → 高级 stage run → 低级 job 的记录、依赖和状态接口。
- `lib/artifacts.py`：外部输入及运行产物的来源、精确路径和内容清单。
- `lib/common.py`：ID、时间、JSON、指纹、原子写入和记录锁。
- `lib/tune_executor.py`：校验冻结 tune job、限定设备并执行一次训练，保存日志和运行证据、登记 adapter 与完整 checkpoint。
- `run_tune_job.py`：单 job 命令行入口；不调度同一高级 run 的其他 job。
- `lib/infer_executor.py`：核对冻结 infer job 和输入来源，分配获准设备、启动一个 worker，并登记逐 task prediction 和运行汇总。
- `lib/infer_worker.py`：一个 Adapter/base 只创建一个 vLLM engine，复用既有 COSINE 推理函数顺序完成该 job 的全部测试 task。
- `run_infer_job.py`：单个 Adapter/base 推理任务入口；不调度同一高级 run 的其他 adapter。
- `lib/infer_orchestrator.py`：核对冻结 infer 计划、解析同一实验的成功 tune adapter，调度各 Adapter/base job，管理资源、失败、中断和逐 task 产物汇总。
- `run_infer.py`：高级 infer 入口；执行已有冻结 infer run，覆盖全部 setting，只执行推理阶段。
- `lib/eval_executor.py`：核对冻结 eval job、成功 infer 来源和精确 GT，绑定全部 prediction manifests，限定设备并登记逐对及整体评测产物。
- `lib/eval_worker.py`：预检全部 prediction/GT 对，单次加载 Qwen/vLLM judge，复用既有 COSINE 柔性判分并保存 raw、scored 和 metrics。
- `run_eval_job.py`：低级批量 eval 入口；执行一个 eval-judge-all job，处理该 setting 全部预测/测试 task 对。
- `lib/tune_orchestrator.py`：调度 Multitask/Singletask 的独立 jobs、CL 串行链和 Transfer 两轮分支，按有效资源分配 GPU 组、管理失败和中断、汇总状态与 adapter 来源。
- `run_tune.py`：高级 tune 入口；从 YAML 新建实验，或执行已有 tune run。支持 Multitask、Singletask、CL 和两种 Transfer setting，只执行训练阶段。
- `pipeline.py`：根据 setting 生成计划，或创建计划及配置记录。
- `requirements.txt`：配置层的 PyYAML 依赖，不替代各模型环境的依赖。

- `lib/eval_orchestrator.py`：核对完整冻结 eval 计划，自动绑定同一实验成功 infer 产物，单次调度 judge，汇总结果表和 setting 矩阵。
- `run_eval.py`：高级 eval 入口；执行已有冻结 eval run，不自动启动 tune 或 infer。

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

## 低级 Adapter/base infer 入口

run_infer_job.py 只执行一个已经 prepare 的冻结 infer job。一个 job 绑定一个成功 tune job 的 final adapter；Zero-shot 不绑定 adapter。入口不调用旧推理 CLI，不重新读取当前 YAML，也不从数据注册表或最新 checkpoint 猜测输入。基础模型、processor 和每个 task 的 test 标注按冻结 input_specs 生成或复核 manifest；非 Zero-shot 必须显式传入已登记的 adapter manifest。

以下示例只说明用法，本轮未执行；工作目录为仓库根目录：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_infer_job.py --job-dir /home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/infer/<infer-run-id>/jobs/infer-whole --gpu-group-index 0 --input adapter=/home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/tune/<tune-run-id>/jobs/tune-whole/artifacts/adapter.json
~~~

Zero-shot 使用其 infer-base job 目录，不传 adapter。--input ROLE=/绝对路径/manifest.json 可按角色重复；已经绑定 inputs 的 job 不接受再次传入替换。adapter 必须来自同一 experiment 内冻结 requires 指定的 tune job，且 producer 已成功、manifest 已登记、slot/task、base 指纹、训练配置及 lineage 一致。不会将 adapter 合并进 base 或叠加多个 adapter。

模型与生成参数全部使用 job 的有效 infer.params：batch_size、dtype、tensor_parallel_size、显存利用率、上下文/生成长度、采样开关、temperature、top_p 和 seed。图像 min/max pixels、视频 min/max pixels、min/max frames 和 FPS 显式应用到选定 processor，并保存临时模型视图供 vLLM 使用；原模型和 processor 文件不改写。视频由既有媒体函数抽帧后送入 vLLM，避免二次抽帧。每个 task 使用完整 test 标注，不增加样本数或批次数的隐式限制。

worker 先预检该 job 全部 task，合并所需媒体数量上限，再创建一次 vLLM engine 和至多一个 LoRA request。按冻结的 tasks 顺序复用同一 engine 完成全部数据；每个 task 内按 batch_size 批量生成。多个 adapter 是多个独立低级 job，可由高级 run_infer.py 自动派发。完成后关闭引擎并清理临时模型视图。

--gpu-group-index 选择冻结 infer.run.gpu_groups 的零起始索引；组内卡数必须等于 tensor_parallel_size。只接受物理 1、2、3，启动 CUDA 前核对 NVML UUID 和 PCI 顺序，并在 worker 再核对设备身份。推理进程使用冻结的 infer.run.python，执行时该 Python 环境需具有既有 Qwen3-VL/vLLM 推理依赖。GPU 文件锁与 tune 共用，阶段槽位按父高级 infer run 的 max_parallel_jobs 限制；低级入口不承担整个高级 run 的 fail_fast 调度。

输出结构如下，根目录为 <infer_root>/<experiment_id>/<run_id>/<job_id>/：

~~~text
execution.json
inference.request.json
logs/infer.log
predictions/<task>.jsonl
inference_result.json
~~~

prediction 沿用既有 COSINE JSONL 格式，保留 prediction、ground_truth、dataset、样本索引/ID、完整 _meta 和 generation。每个 task 完成后核对全量条数、顺序及 GT/元信息；空回答统计为 empty_predictions，交给后续 eval 处理。全部 task 完成且输入指纹未变化后，在 job 的 artifacts/ 登记 prediction-<task>.json 和 inference-summary.json，随后将 job 标为 succeeded。

失败或 SIGINT/SIGTERM 时保留日志、已生成文件和执行证据，清理该 worker 进程组，并将 job 标为 failed/cancelled；worker 已进入执行流程时另写 inference_failure.json。已有输出目录、执行收据、终态或正在执行的 job 会被拒绝，不覆盖、不自动续写部分预测或重试。低级入口只将父 infer run 推进为 running，整体完成状态由高级编排汇总。高级 eval 可通过 run_eval.py 独立启动；最终 YAML 参数核对仍留待后续阶段。

## 低级批量 Qwen 柔性 eval 入口

run_eval_job.py 执行已经 prepare 的单个 eval-judge-all 冻结 job。该 job 一次接收本 setting 的全部 prediction/task 对：Zero-shot、Multitask、Singletask 各 6 对，CL 36 对，Transfer-round1/round2 各 30 对。启动一个 worker，全部对共用一次 Qwen/vLLM engine 和一次 judge 权重加载；按冻结对顺序处理，每对内部使用 batch_size 批量评测。低级入口不负责自动搜集整个高级 run 的输入或结束父 run；高级 run_eval.py 负责搜集冻结计划的全部输入并汇总父 run。

须按 --input ROLE=/绝对路径/manifest.json 显式传入全部 prediction manifests，角色名不得重复。入口按 producer、slot 和 task 对应冻结 requires；不依赖角色名猜测测试集。每份预测必须由同一 experiment 唯一登记且已成功的 infer run 内对应成功 job 产生，manifest 已登记，完整预测、执行收据、一次模型加载和输入来源均需复核。非 Zero-shot producer 必须使用高级 infer 的标准 adapter 输入角色；其他自定义 adapter 角色当前不接受。judge-model、judge-processor 以及六 task 的 test annotation 按冻结 input_specs 自动生成或复核 manifest；GT 必须与 infer 当时绑定的 annotation manifest 完全一致。已绑定 inputs 不接受再次传参替换。

以下仅为命令格式示例，本轮未执行。必须补齐该冻结 job 全部 scoring_pairs 对应的 --input，单独传入此处展示的一个 prediction 不能启动六 task 的完整评测：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_eval_job.py \
  --job-dir /home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/eval/<eval-run-id>/jobs/eval-judge-all \
  --gpu-group-index 0 \
  --input prediction-motion=/home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/infer/<infer-run-id>/jobs/infer-base/artifacts/prediction-motion.json
~~~

worker 先对全部预测与精确 GT 做全量条数、顺序、ID、元信息和 prompt 预检，再创建共享 judge engine。不重新生成答案，不读媒体，不按样本数隐式截取，不增加基于正则的答案匹配。沿用 evaluation/COSINE 的既有 prompt 与计分：MCQ/OPEN 交由 Qwen 语义判断；MRA 由 Qwen 抽取数值后计算既有相对准确率；invalid 和解析失败计 0 且保留在分母。超长 judge prompt 按既有行为报错，不静默截断。

模型与运行参数读取该 job 的冻结 eval.params/run：batch_size、dtype、tensor_parallel_size、gpu_memory_utilization、max_model_len、max_new_tokens、repair_attempts、seed 和 python，不重新读取后来修改的 YAML。--gpu-group-index 为冻结 eval.run.gpu_groups 的零起始索引，组内卡数须等于 tensor_parallel_size；只允许物理 GPU 1–3，复用 tune/infer 的物理设备锁和 eval 父 run 的槽位锁，启动前及 worker 内分别核对实际设备身份。

输出根目录为 <eval_root>/<experiment_id>/<eval-run-id>/eval-judge-all/：

~~~text
execution.json
evaluation.request.json
logs/eval.log
pairs/<prediction-job>--<task>/judge.raw.jsonl
pairs/<prediction-job>--<task>/judge.scored.jsonl
pairs/<prediction-job>--<task>/judge.metrics.json
judge.metrics.json
evaluation_result.json
~~~

逐对保存原始 judge 文本、解析/计分结果和 metrics，保留 prediction producer、task、输入指纹及 judge 配置来源。关闭 engine 后逐对从 raw 重建 scored/metrics，并核对输入未变化，再登记 judge_raw、judge_scored、逐对辅助 summary 和整体 summary；根级 judge.metrics.json 是唯一主 metrics，slot 为 combined。主指标为 combined_flat_rawacc，整体按全部预测/样本对加权；不同 adapter 对同一测试样本的结果分别保留，不去重，不把逐 task 的百分比直接取平均。

失败或 SIGINT/SIGTERM 时保留已有文件、日志和执行证据，清理 worker 进程组，并将 job 记为 failed/cancelled；worker 已进入执行流程时另写 evaluation_failure.json。拒绝已有输出、执行收据和终态 job，不覆盖、不自动续写或重试。成功只在全部计划对与整体产物完整登记后记录；低级执行器仅将父 eval run 推进到 running，父 run/experiment 的最终汇总由高级 run_eval.py 编排负责。CLI 成功返回 0、失败 1、中断 130。

## 高级 infer 入口与资源调度

先通过 pipeline.py prepare 或高级 tune 入口创建实验记录。训练型 setting 完成同一 experiment 的高级 tune run 后，独立启动其 infer run；Zero-shot 直接使用 prepare 创建的 infer run。高级入口只接受既有冻结 run，不重新读取当前 YAML、不创建新实验，也不自动启动 tune/eval。例如（仅提供用法，本轮未执行）：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_infer.py --run-dir /home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/infer/<infer-run-id>
~~~

高级进程不加载模型；每个 Adapter/base 只启动一个 run_infer_job.py，在一个 vLLM engine 中完成该 job 全部 task。Zero-shot 一个 base job 测六 task；Multitask 一个 whole adapter job 测六 task；Singletask 六个 adapter 分别测对应 task；CL 按冻结训练顺序派发六个 adapter，各测六 task，合计 36 份预测。两种 Transfer 各使用三十个第二轮 adapter；round1 测第一轮 source task，round2 测第二轮 target task。全部测试 task 都使用完整 test 标注。

非 Zero-shot 要求同一 experiment 唯一登记的 tune run 已成功，并按每个 infer job 的 requires、adapter_ref 和 slot 精确查找成功 producer 的 final adapter manifest；自动传给低级入口。不会搜索最新 checkpoint、跨实验取权重或把其他 adapter 当成替代品。Zero-shot 不绑定 adapter。

- 高级 run 的 infer.run.max_parallel_jobs 和 fail_fast 控制整个 run 的并发与失败策略。每个 job 的 infer.params、infer.run.python 和 gpu_groups 使用自身冻结有效配置；高级入口不覆写局部设置。调整 YAML 后需重新 prepare 实验，已有 run 始终读取原快照。
- 调度器按各 job 的有效 GPU 组和 tensor_parallel_size 选择本 run 内空闲的物理设备，不重叠派发；仅允许物理 GPU 1–3。低级执行器继续持有与 tune 共用的 GPU 锁及本 infer run 的槽位锁，启动 worker 前核对设备身份。max_parallel_jobs 为 1 或只有一组设备时顺序执行；多个独立组可并行处理不同 adapter。CL 的推理不要求串行继承，六个已完成 adapter 可独立推理。
- fail_fast 为 true 时，发现失败即停止派发，请求其他活动低级入口清理退出，未启动 job 标为 cancelled；为 false 时继续其他独立 job，最终有任何失败则高级 run 失败。若低级退出但清理收据缺失/损坏、cleanup_completed 未确认或记录清理错误，则不论 fail_fast 都停止整个 run，保留证据，避免复用清理未确认的设备。
- SIGINT/SIGTERM 停止派发并等待低级入口关闭其 worker 和临时模型视图；保留已成功预测及失败证据，整体记为 cancelled。高级层不强杀仍在清理的低级进程。

可接管 planned/running 高级 run 中未执行的 planned jobs 和已完成的 succeeded jobs；已成功 job 只有在输入、adapter 来源、执行收据、一次模型加载、完整逐 task 预测和汇总都通过只读复核后才跳过。running、failed/cancelled job，或已有 inputs、执行收据/输出的 planned job 会被拒绝，需先人工审查。不自动重试、不续写部分预测、不覆盖终态；不要同时通过低级入口启动同一 run 的 job。

每次高级调用在记录树的 orchestrations/<invocation-id>/ 保存 execution.json、各低级入口的 <job-id>.log 及 summary.json；infer-summary.json 保存本次总体汇总。汇总记录 job/task 顺序、状态、退出码、设备分配、输出目录、日志、adapter manifest/producer/slot/指纹及逐 task prediction/summary manifest。只有全部计划 job 成功且完整预测产物复核通过，高级 infer run 才记为 succeeded；产物复核失败不会改写已经终态的子 job，但高级 run 会失败并记录原因。

失败或中断仅终结本 infer run，不提前结束 experiment；全部已登记阶段均成功时 experiment 才成功，默认启用 eval 的实验在 infer 完成后继续保持 active。高级 CLI 成功返回 0、失败返回 1、中断返回 130。高级 eval 可通过 run_eval.py 独立启动；最终 YAML 参数核对仍留给后续阶段。

## 高级 eval 编排与结果汇总

完成同一 experiment 的 infer run 后，使用已 prepare 的冻结 eval run 目录启动评测。入口只接受绝对路径，不重新读取当前 YAML、不创建新实验，不自动执行训练或推理。例如（仅说明用法，本轮未执行）：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_eval.py --run-dir /home/linruikang/data/experiment-runs/Qwen3VL-LoRA-FT/<experiment-id>/stage-runs/eval/<eval-run-id>
~~~

高级入口核对 experiment/run/job 的冻结配置、计划和登记关系，要求同一 experiment 唯一的 infer run 已成功；按 scoring_pairs 顺序绑定对应成功 infer job 的 prediction manifest，复核 producer、slot、task、内容指纹及推理时使用的精确 test GT。缺失、重复、跨实验、错位或被修改的预测会被拒绝，不搜索其他结果作为替代。

一个高级 eval run 只有一个 eval-judge-all job。全部 prediction manifests 通过重复的 --input 参数一次传给 run_eval_job.py；低级执行器只加载一次 Qwen/vLLM judge，按 batch 完成全部评测。高级进程不加载模型，不改变既有 COSINE prompt 与计分规则。Zero-shot、Multitask、Singletask 各 6 对，CL 36 对，两种 Transfer 各 30 对。

运行参数来自 job 的冻结 eval.params/run。唯一 judge job 固定选择其 eval.run.gpu_groups 的第一组，低级执行器等待与 tune/infer 共用的设备锁和 eval 槽位锁；只允许物理 GPU 1–3，不查询设备负载，不自动换卡。max_parallel_jobs 与 fail_fast 继续记录，但不会把唯一 job 拆成多个 judge 或多次加载模型。

可接管 planned/running 高级 run 中未执行的 planned job 或已经成功的 succeeded job。成功 job 必须通过完整输入、执行请求、清理收据、逐对 raw/scored/metrics、整体指标及登记 manifest 的只读复核，才直接汇总并跳过模型启动。running、failed/cancelled job，或已有 inputs、收据、请求或输出的 planned job 会被拒绝；不自动重试、不续写、不覆盖已有评测。不要同时通过低级入口启动同一 run 的 job。

每次调用在记录树的 orchestrations/<invocation-id>/ 保存 execution.json、eval-judge-all.log（实际启动低级 job 时）、summary.json，以及成功后生成的 results.json、results.csv、matrix.json、matrix.csv。run 根目录 eval-summary.json 保存总体状态、scoring_pairs、来源 infer run、低级 job 状态/设备/日志/产物，以及这些结果文件的绝对路径和 SHA-256。results.json 保留逐对指标、预测/GT/adapter 来源和指纹；results.csv 提供同序扁平表。底层 raw/scored/metrics 仍在原 eval payload 下。

矩阵以 combined_flat_rawacc 的 0–1 比例保存，不转换为百分比：

| Setting | 行 | 列 | 有效格数 |
| --- | --- | --- | --- |
| Zero-shot | base | 六个测试 task | 6 |
| Multitask | whole adapter | 六个测试 task | 6 |
| Singletask | 六个独立 task adapter | 六个测试 task，仅对应对角线 | 6 |
| CL（包括 easy2difficult） | 按冻结 task_order 排列的六阶段 adapter | 六个测试 task | 36 |
| Transfer-round1 | 第一轮 source task | 第二轮训练 task；每格测试 source task | 30 |
| Transfer-round2 | 第一轮 source task | 第二轮训练 task；每格测试第二轮 task | 30 |

列及非 CL 行的 task 顺序为 motion、hypo、pattern、nav、view、geo。未安排测试的格在 JSON 为 null、CSV 为空，不能当成 0 分；matrix.json 同时保存 sample_counts 与 pair_ids，区分未测和空数据。每条结果保留实际 test_task、CL 阶段或 Transfer source/second_round，避免混淆两种 Transfer 测试口径。

整体 combined_flat_rawacc 直接沿用从全部 raw judgments 重建并复核的低级整体 metrics：所有 prediction/sample 对的 score_sum / total，不平均 task 百分比；不同 adapter 测同一样本仍分别计数。invalid 和解析失败保留在分母；空数据的 null 沿用既有计分定义。不会额外计算未经约定的遗忘率或迁移指标。

SIGINT/SIGTERM 请求低级入口清理其 worker 并等待退出；高级层不强杀仍在关闭 vLLM 的进程。失败/中断保留低级证据，记录高级 run 为 failed/cancelled，不改写已成功子 job。只有所有计划 pair、清理证据及最终表格完整时，高级 run 才 succeeded；全部登记阶段均成功后 experiment 才 succeeded。失败不提前结束 experiment；CLI 返回码为成功 0、失败 1、中断 130。最终默认/setting YAML 参数核对仍为后续独立步骤。

## 高级 tune 入口与资源调度

`run_tune.py` 将一个 tune 高级 run 的任务交给多个低级 `run_tune_job.py` 进程；高级进程不加载模型。Multitask 只调度 `tune-whole`，使用 whole/train；Singletask 按 motion、hypo、pattern、nav、view、geo 的顺序放入队列，共六个 `tune-<task>`。每个任务从相同原始 base model 新建独立 LoRA、optimizer、scheduler、epoch 和 global step，不继承其他 task 的 adapter。该队列顺序不构成持续学习关系，并行任务的结束顺序可以不同。CL 则按冻结的 task_order 串行训练六个阶段，具体继承关系见下文。

从 setting YAML 创建新 experiment、冻结所有已启用阶段的计划及 job，然后只执行 tune。例如（本轮仅提供命令，未执行）：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py \
  --config scripts/experiments/configs/multitask.yaml \
  --experiment-id multitask-example

/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py \
  --config scripts/experiments/configs/singletask.yaml \
  --experiment-id singletask-example
~~~

`--default`、`--set` 的含义与 `pipeline.py prepare` 一致；未指定值沿用默认 YAML。已有 experiment ID 不覆盖，重新实验使用新 ID。入口在创建记录前拒绝 Zero-shot 及未启用 tune 的配置。infer/eval 即使已登记，也只保持 planned。

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
- `max_parallel_jobs: 1` 或只有一组 GPU 时依次训练。独立任务或 Transfer 分支配置多组独立 GPU 并提高并发上限后可并行；CL 的任务依赖仍强制六阶段串行。单组可含多张 GPU，由低级执行器使用 torchrun。
- `fail_fast: true` 在发现首个失败后停止派发，并请求其他活动低级进程退出、等待其关闭训练进程及登记已完整写出的 checkpoint；未启动任务标为 cancelled。`false` 继续其他独立任务，最后仍将有失败的高级 run 标为 failed；CL 后续阶段依赖前一阶段，Transfer 第二轮依赖各自的第一轮来源，即使关闭 fail_fast，依赖失败或 adapter 审计失败后也会取消受影响的后续任务。如果低级进程已经退出而 job 仍为 running，或其执行收据记录清理失败，调度器停止整个 run，保留失败证据供检查残留训练进程，不继续复用该设备。
- SIGINT/SIGTERM 同样停止派发、请求低级清理并等待退出，保留已有成功产物与证据，整体记为 cancelled。高级层不强杀仍在清理或保存 checkpoint 的低级进程。

每次高级调用在 run 记录树内新建 `orchestrations/<invocation-id>/`，保存 `execution.json`、各低级入口的 `<job-id>.log` 和 `summary.json`；低级详细训练日志仍位于原 payload 的 `logs/train.log`。run 根目录的 `tune-summary.json` 保存最新汇总，列出 job_order/task_order、每个 job 的状态、退出码、GPU/端口分配、日志、输出目录及 adapter manifest 路径/指纹/producer/slot，并记录 depends_on、lineage、training_state_policy、是否显式恢复、父 adapter 的 manifest/指纹/producer 和依赖阻断原因。若 adapter 内容或来源复核失败，高级 run 不记录成功，即使该 job 先前已是 succeeded。

所有计划训练任务成功且 adapter 完整后，tune run 才变为 succeeded。本入口将失败或中断记录在 tune run/job 层，不提前结束整个 experiment；experiment 仅在所有已启用阶段均成功时记为 succeeded，默认启用 infer/eval 的实验训练完成后仍为 active。高级 CLI 成功返回 0，失败返回 1，中断返回 130。训练后可独立调用高级 infer 入口；eval 可通过高级 run_eval.py 独立执行；自动重试记录和最终 YAML 参数核对继续留待后续阶段。

## Transfer 两轮迁移高级 tune

Transfer-round1 和 Transfer-round2 使用相同的训练结构：第一轮在 motion、hypo、pattern、nav、view、geo 上分别从 base model 新建 LoRA，共六个独立 job；第二轮将每个第一轮 adapter 分别迁移到另外五个 task，共三十个独立 job。每个高级 tune run 合计 36 个低级训练 job。

- 第一轮 job：`tune-round1-<task>`，输出 slot 为 `round1-<task>`。
- 第二轮 job：`tune-round2-<source>-to-<target>`，其中 source 与 target 不同，训练 target/train，输出 slot 为 `round2-<source>-to-<target>`。
- 第二轮只依赖同一高级 run 的 `tune-round1-<source>`。同一 source 的五个子 job 都从该第一轮 final adapter 独立开始，不串联彼此的训练结果。

第二轮继续训练父 LoRA 参数，重新初始化 optimizer、scheduler、epoch 和 global step；不将 LoRA 合并到 base，不额外叠加 adapter，也不沿用父训练状态。只有显式的同 task checkpoint 恢复才使用完整训练状态，继续沿用低级入口的恢复规则。

高级入口核对冻结计划的全部 36 个 job、排队顺序、task、依赖、requires、lineage、输出 slot 及模型/数据输入。第二轮启动前须等待其第一轮低级进程退出、job 成功且 final adapter 审计通过，再自动绑定该父 job 的绝对 manifest 路径。已有绑定必须与登记的父产物完全一致，不能换成其他 source、第二轮兄弟 job、其他 run 或最新 checkpoint。第一轮和第二轮的成功产物均核对轮次 lineage、训练状态策略与继承输入指纹。

队列先列全部第一轮 job，再按 source/target 列第二轮 job；资源满足时按队列选择已就绪的任务。第二轮只等待自身父 job，不要求其他来源的第一轮全部结束。并发数、job 局部 GPU 分组、端口冲突、资源锁和中断清理均使用既有高级调度机制。父 job 失败或 adapter 审计失败时，其五个第二轮 job 会被阻断；`fail_fast: false` 可继续其余独立来源分支，`true` 则停止整个 run 的派发并清理活动进程。

以下是使用示例，本轮未执行：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py --config scripts/experiments/configs/transfer_round1.yaml --experiment-id transfer-round1-example

/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py --config scripts/experiments/configs/transfer_round2.yaml --experiment-id transfer-round2-example
~~~

通过 `--run-dir` 接管已有 planned/running 高级 run 时，已成功的第二轮 job 必须具有成功且通过复核的第一轮父 job 和精确父绑定；不同来源分支可以具有不同的完成进度。其余 job 状态、拒绝重复执行与显式恢复规则沿用前述高级入口，不自动重置终态或创建 retry。

36 个 final adapter 分别保存在 `<checkpoints_root>/<experiment_id>/<run_id>/<job_id>/adapter/`，第一轮六个 adapter 保留供所有子 job 读取，子 job 不覆盖父产物。`tune-summary.json` 记录每个 job 的轮次、source/train task、父 adapter、状态、资源、输出和阻断原因；全部 36 个 job 成功且产物核对通过后，高级 tune run 才记为 succeeded。

两个 setting 的区别在后续测试选择：Transfer-round1 将三十个第二轮 adapter 分别测试其第一轮 source task；Transfer-round2 测试其第二轮 target task。现有计划已保留该区别，本入口只执行 tune，infer/eval 仍保持 planned。默认 YAML 与 setting YAML 的参数核对留待最终阶段。

## CL 持续学习高级 tune

CL-MHPNVG（easy2difficult）使用 configs/cl_mhpnvg.yaml，默认顺序为 motion → hypo → pattern → nav → view → geo。自定义 CL 使用 configs/cl_custom.yaml，在 experiment.task_order 填入六个 task 的完整排列；每个 task 必须且只能出现一次。两者均通过 setting: cl 使用同一高级入口。例如（仅为使用示例，本轮未执行）：

~~~bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py --config scripts/experiments/configs/cl_mhpnvg.yaml --experiment-id cl-mhpnvg-example

/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python scripts/experiments/run_tune.py --config scripts/experiments/configs/cl_custom.yaml --experiment-id cl-custom-example
~~~

高级入口将顺序冻结为 tune-01-<task> 至 tune-06-<task> 六个低级 job。第一阶段从原始 base model 新建 LoRA；第二阶段起继续训练前一阶段的同一个 LoRA，只继承其参数，每阶段重新初始化 optimizer、scheduler、epoch 和 global step。不会把 adapter 合入 base model，不会额外叠加一个新 adapter，也不会跨 task 继承完整训练状态。各 job 的训练参数和 GPU 分组仍按既有默认 YAML → setting → job → CLI 的优先级解析。

调度器等待父低级执行器退出、job 成功且最终 adapter 的内容与登记来源核对通过后，才启动下一阶段；低级上下文检查也延迟到该依赖就绪时执行。未绑定输入时自动传入 --input parent-<task>=<父 job 登记的绝对 adapter manifest 路径>；已有绑定则核对它与同一 experiment、同一 tune run 的前一 job 产物完全一致，不重复传入或覆盖。检查包含 producer、slot、配置指纹、lineage、训练状态策略和继承输入指纹，不能用其他 task、其他 run 的 adapter 或最新 checkpoint 替换。

即使 max_parallel_jobs 大于 1、配置了多组 GPU，单条 CL 链也只会有一个活动训练 job。前一阶段失败、adapter 审计失败或启动前继承检查失败时，不启动其后的阶段；fail_fast: false 也不能绕过依赖。SIGINT/SIGTERM 沿用高级停止和低级清理流程。已经成功的 adapter 保留，未启动的后续阶段记为 cancelled；失败和审计原因保存在汇总 details，依赖阻断的 job 另外记录 blocked_by。

--run-dir 可以接管仍为 planned/running 的高级 run，但已成功的 CL 阶段必须组成连续前缀；跳过之前会复核最终 adapter 和父权重绑定关系。非连续成功记录、错绑父 adapter、尚未成功的 producer 或已开始/终态失败的 job 均不会被当成新的训练入口。显式同 task 的完整 checkpoint 恢复仍由低级入口处理；恢复成功后可接管其余 planned 阶段，高级调度器不自动寻找 checkpoint 或创建 retry。

六个阶段的最终 adapter 分别保留在各自的 <checkpoints_root>/<experiment_id>/<run_id>/<job_id>/adapter/，后续训练不会覆盖前一阶段的权重。tune-summary.json 按训练顺序给出全部六个 adapter 及父子来源链，供后续 infer/eval 使用；现有计划仍是六个 adapter 分别测试六个 task，共 36 对结果，本轮不执行 infer/eval，也不调整 YAML 参数。
