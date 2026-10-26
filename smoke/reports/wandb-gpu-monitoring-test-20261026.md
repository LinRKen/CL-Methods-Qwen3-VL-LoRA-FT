# W&B GPU 监控 test 分支验证记录

- 时间：2026-10-26（Asia/Shanghai）
- 仓库：`/home/linruikang/CL/CL-Methods/Qwen3-VL/LoRA-FT`
- 分支：`test/wandb-gpu-monitoring`
- 被测提交：`d6b052a3236bc08259cfd4a974a27784649d68ee`
- 被测实现：`qwen-vl-finetune/qwenvl/train/wandb_utils.py` 与 `qwen-vl-finetune/qwenvl/train/train_qwen.py`
- 验证代码：`smoke/scripts/`

## 验证范围

本轮只在 `smoke/` 下增加验证脚本和报告，没有修改主仓库源码。验证覆盖：

1. GPU 选择、UUID 规范化、物理 GPU 0 拒绝、重复和非法输入拒绝。
2. W&B 启用和禁用时的环境变量配置、旧配置覆盖、run 初始化顺序及非主进程行为。
3. 生产训练入口中 GPU 解析、参数解析、W&B 配置和 Trainer 创建的调用顺序。
4. 使用生产 helper、Transformers `Trainer` 和原生 `WandbCallback` 的真实短时训练。
5. 单卡和双卡 DDP 下的 GPU 身份继承、W&B run 归属以及原生系统指标筛选。

## 环境

- Python：`/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python`
- PyTorch：2.6.0+cu124
- Transformers：4.57.3
- Accelerate：1.7.0
- W&B：0.31.0
- W&B 使用 offline 模式；每次训练 24 steps，batch/GPU=16，学习率 1e-4。

## 结果

### CPU 检查

`test_wandb_gpu_monitoring.py` 共 15 项测试全部通过，耗时 0.529 秒。物理 GPU 0 的相关分支使用 mock，不创建 CUDA 上下文。

### 物理 GPU 2 单卡

执行：

```bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B \
  smoke/scripts/run_wandb_gpu_monitoring.py \
  --devices 2 \
  --result-dir smoke/results/wandb-gpu-monitoring-test-20261026.single-gpu2
```

- 启动器退出码：0。
- 训练步数：24/24。
- `train_loss`：0.032606822996361494。
- 模型参数发生更新：是。
- W&B run：`g3ug6075`，offline，只有 rank 0 创建。
- W&B GPU 筛选：`[2]`。
- 进程内 CUDA 设备：`cuda:0`，对应物理 GPU 2。
- 观察到的系统指标 GPU 集合：`{2}`。
- 显存、功耗、温度、利用率：均为 10 次原生采样。

### 物理 GPU 2、3 双卡 DDP

执行：

```bash
/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python -B \
  smoke/scripts/run_wandb_gpu_monitoring.py \
  --devices 2,3 \
  --result-dir smoke/results/wandb-gpu-monitoring-test-20261026.ddp-gpu23
```

- 启动器退出码：0。
- rank 0/1 均完成 24/24 步。
- `train_loss`：0.030121029456495307。
- 两个 rank 的模型参数均发生更新。
- W&B run：`w0vls257`，offline，仅 rank 0 创建。
- W&B GPU 筛选：`[2, 3]`。
- rank 0 的 `cuda:0` 对应物理 GPU 2；rank 1 的 `cuda:1` 对应物理 GPU 3。
- 两个 worker 均继承相同的 UUID 形式 `CUDA_VISIBLE_DEVICES`。
- 观察到的系统指标 GPU 集合：`{2, 3}`。
- 物理 GPU 2、3 的显存、功耗、温度、利用率均为 10 次原生采样。

## 结论

本轮验证通过。当前实现能够从唯一的 `CUDA_VISIBLE_DEVICES` 配置中确定训练设备，并让 W&B 原生系统监控只采集相同的物理 GPU。单卡和双卡 DDP 均保持单个主进程 W&B run，物理 GPU 0 未被使用。两次结果均通过 `--check-only` 复核。

## 结果位置

完整运行证据保存在被忽略的外部结果目录：

- `smoke/results/wandb-gpu-monitoring-test-20261026.single-gpu2/`
- `smoke/results/wandb-gpu-monitoring-test-20261026.ddp-gpu23/`

## 限制

本轮使用合成小网络和 Transformers `Trainer` 验证生产 helper 与 W&B callback 的接入；没有运行完整 Qwen3-VL、COSINE 数据集、LoRA 微调、DeepSpeed 或云端 W&B 页面验证。
