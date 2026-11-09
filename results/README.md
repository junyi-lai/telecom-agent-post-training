# 最终结果目录

`results/` 保存可复用的训练数据、原始采集结果和经过确认的最终评测报告。

## 目录说明

- `results/data/processed/`：最终基线、教师数据、SFT、RL 和开发 Prompt。
- `results/reports/`：最终运行摘要和数据审计报告。
- `results/runs/baseline/`：最终 656 条基线原始结果。
- `results/runs/teacher/`：最终 310 条教师原始结果。
- `results/runs/smoke_accepted/`：通过验收的学生链路 smoke。
- `results/runs/teacher_smoke_accepted/`：通过验收的教师链路 smoke。

基线、教师轨迹、SFT/RL 数据、任务清单、校验文件和最终评测报告进入 Git。模型权重、训练 checkpoint、评测原始轨迹、日志、smoke 运行目录和缓存不进入 Git。

## 当前结果

- 基线：86 / 656 严格成功，13.1%。
- 教师：302 / 310 严格成功，97.4%。
- SFT 普通组：227 行、159,949 assistant token、164 个任务。
- SFT 失败感知组：225 行、159,960 assistant token、164 个任务。
- RL 训练 Prompt：普通组 164 行，失败感知组 328 行。
- RL 开发 Prompt：普通组 10 行，失败感知组 20 行。

五模型三 seed 冻结测试已完成，共 600 条有效轨迹。成功率分别为 base 80.00%、uniform SFT 55.00%、failure-aware SFT 75.83%、uniform GRPO 72.50%、failure-aware GRPO 88.33%。完整任务级区间与限制见[最终评测报告](reports/evaluation_three_seeds_20261107.md)。
