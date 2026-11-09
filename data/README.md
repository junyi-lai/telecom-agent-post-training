# 数据使用说明

仓库提供基线采集、教师纠错和训练数据的完整可复用版本。读者可以跳过需要外部模型与付费 API 的数据采集阶段，直接运行 SFT 或 GRPO。

## 包含的数据

| 文件 | 规模 | 用途 |
| --- | ---: | --- |
| `results/runs/baseline/results.json` | 656 条轨迹 | 学生模型在 164 个训练任务上的原始基线结果 |
| `results/runs/teacher/results.json` | 310 条轨迹 | 离线教师对 155 个失败任务的纠错结果 |
| `results/runs/teacher/teacher_task_ids.json` | 155 个任务 | 教师采集覆盖的任务清单 |
| `results/data/processed/train.jsonl` | 656 行 | 归一化基线轨迹与失败标签 |
| `results/data/processed/teacher.jsonl` | 310 行 | 归一化教师轨迹与来源信息 |
| `results/data/processed/sft/sft_uniform.jsonl` | 227 行 | 普通 SFT 对话 |
| `results/data/processed/sft/sft_failure_aware.jsonl` | 225 行 | 失败感知 SFT 对话 |
| `results/data/processed/rl/rl_uniform.parquet` | 164 行 | 普通 GRPO Prompt |
| `results/data/processed/rl/rl_failure_aware.parquet` | 328 行 | 失败感知 GRPO Prompt |
| `results/data/processed/rl_dev/` | 10 / 20 行 | 两组开发 Prompt |

SFT 同时提供 JSONL 和 Parquet。数据构建摘要位于对应目录的 `dataset_summary.json` 或 `rl_dataset_summary.json`。

## 直接训练

准备好 Python、CUDA、GPU 训练依赖和学生模型后：

```bash
bash scripts/setup_inference.sh
make download-model
bash scripts/setup_training.sh
ARM=uniform make train-sft
ARM=failure_aware make train-sft
ARM=uniform make train-grpo
ARM=failure_aware make train-grpo
```

训练脚本直接读取 `results/data/processed/`，不需要先调用 `make collect` 或 `make teacher`。

## 从原始数据重建

如需重新执行归一化、SFT 构建或 RL Prompt 构建：

```bash
make normalize
make normalize-teacher
make dataset
make rl-dataset
```

如需完整重跑基线、教师和用户模拟器，执行 `make collect` 与 `make teacher`。这些命令会调用 GPU 和付费 API。

## 校验

在仓库根目录执行：

```bash
sha256sum -c data/SHA256SUMS
```

`SHA256SUMS` 覆盖公开的原始采集结果、归一化轨迹、SFT/RL 数据和任务清单，用于发现下载损坏或误改。

## 数据来源与格式

- 任务场景来自固定版本的 tau2-bench 电信域；
- 基线由 Qwen3.5-2B 生成；
- 教师和用户模拟器使用 DeepSeek Flash；
- 任务 ID、消息、工具调用、关系奖励、失败标签和模型 provenance 均保留；
- 示例手机号、账户和客户信息来自基准环境的合成数据。

项目创建的数据整理与格式转换适用仓库的 Apache-2.0 许可；tau2-bench 上游内容和远程模型生成内容仍受各自原始许可与服务条款约束。

评测原始轨迹、模型权重、checkpoint、日志和中间缓存不在公开数据包内。
