# 五模型冻结测试：seed 300/301/302

## 评测范围与校验

每模型使用同一组 40 个冻结任务和 seed 300、301、302，共 120 条轨迹；五模型合计 600 条。本次新增 400 条，保留原 seed 300 的有效结果。逐 seed 校验任务集合、唯一性、记录数和基础设施终止；最终每任务恰有三个 seed，所有接受结果的基础设施错误均为 0。

推理设备显存为 16GB，五模型依次加载，任务并发 8。学生贪心解码，用户模拟器温度 0.7；32K 上下文，vLLM GPU 比例 0.60、max_num_seqs=8。没有重新训练、运行 smoke 或调整模型。

## 严格任务成功率

| 模型 | seed 300 | seed 301 | seed 302 | 三 seed 合计 | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 31/40 | 32/40 | 33/40 | 96/120 | 80.00% |
| uniform SFT | 23/40 | 20/40 | 23/40 | 66/120 | 55.00% |
| failure-aware SFT | 30/40 | 31/40 | 30/40 | 91/120 | 75.83% |
| uniform GRPO | 30/40 | 30/40 | 27/40 | 87/120 | 72.50% |
| failure-aware GRPO | 32/40 | 37/40 | 37/40 | 106/120 | 88.33% |

## 任务级配对比较

先对每任务的三个 seed 求平均，再按相同 40 个任务簇做 10,000 次配对 bootstrap，随机种子 20261103。差值方向为表中左侧模型减去比较对象，单位为百分点；三条轨迹不视为独立样本。区间未进行多重比较校正。

| 比较 | 平均差 | 配对 bootstrap 95% 区间 |
| --- | ---: | ---: |
| uniform SFT − base | −25.00 | [−37.50, −12.50] |
| failure-aware SFT − base | −4.17 | [−13.33, +5.00] |
| uniform GRPO − base | −7.50 | [−17.50, +2.50] |
| failure-aware GRPO − base | +8.33 | [0.00, +17.50] |
| failure-aware GRPO − failure-aware SFT | +12.50 | [+4.17, +21.67] |
| failure-aware GRPO − uniform GRPO | +15.83 | [+6.67, +25.00] |

failure-aware GRPO 在本次冻结测试中表现最好，且相对自身 SFT 起点与 uniform GRPO 的区间均在零以上。相对 base 的区间下界触及零，因此不能据此声称已稳健证明优于 base。uniform SFT 在本测试上明显低于 base；另两组相对 base 的区间跨零。

完整机器可读统计见 `evaluation_three_seeds_20261107.json`。原始轨迹保存在 `tmp/artifacts/eval_seeds301_302_20261107/`，规范化数据在 `tmp/data/eval/`。运行中的别名映射、恢复与资源约束已单独整理在 [项目复盘](../../docs/PROJECT_RETROSPECTIVE.md)。
