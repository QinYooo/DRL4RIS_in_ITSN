# Baseline Comparison Guide

## 概述

本项目现在支持将 DRL 模型与以下 baseline 方法进行比较分析：

1. **Zero-Action Baseline**: 不使用 RIS 相位控制 (Phi = I)
2. **Baseline Optimizer**: 使用 ZF+SDR 优化方法 (baseline/baseline_optimizer.py)
3. **DRL Model**: 训练好的 RecurrentPPO 模型

## 使用方式

### 方法一: 在训练时自动运行 baseline 评估

在训练脚本中添加 `--run-baseline-optimizer` 参数：

```bash
# 训练并运行所有 baseline 评估
python train_recurrentPPO.py \
    --use-ae \
    --ae-checkpoint checkpoints/channel_ae/channel_ae_best.pth \
    --total-timesteps 102400 \
    --run-baseline-optimizer

# 跳过 zero-action baseline (只运行 baseline optimizer)
python train_recurrentPPO.py \
    --use-ae \
    --ae-checkpoint checkpoints/channel_ae/channel_ae_best.pth \
    --total-timesteps 102400 \
    --run-baseline-optimizer \
    --skip-zero-baseline
```

### 方法二: 独立运行比较评估脚本

使用 `evaluate_comparison.py` 对已训练的模型进行评估：

```bash
# 评估所有方法 (zero + baseline + DRL)
python evaluate_comparison.py \
    --model-path logs/RecurrentPPO_AE_xxxxxx/final_model.zip \
    --vecnormalize-path logs/RecurrentPPO_AE_xxxxxx/final_model/vec_normalize.pkl \
    --use-ae \
    --ae-checkpoint checkpoints/channel_ae/channel_ae_best.pth \
    --method all \
    --seed 42 \
    --output-dir comparison_results

# 只评估 baseline optimizer
python evaluate_comparison.py \
    --method baseline \
    --seed 42 \
    --output-dir comparison_results

# 只评估 DRL 模型
python evaluate_comparison.py \
    --model-path logs/RecurrentPPO_AE_xxxxxx/final_model.zip \
    --vecnormalize-path logs/RecurrentPPO_AE_xxxxxx/final_model/vec_normalize.pkl \
    --use-ae \
    --ae-checkpoint checkpoints/channel_ae/channel_ae_best.pth \
    --method drl \
    --seed 42
```

### 方法三: 只使用全几何状态 (不使用 AE)

```bash
# 训练全几何状态模型并运行 baseline 评估
python train_recurrentPPO.py \
    --total-timesteps 102400 \
    --run-baseline-optimizer

# 比较全几何状态模型
python evaluate_comparison.py \
    --model-path logs/RecurrentPPO_Geometry_xxxxxx/final_model.zip \
    --vecnormalize-path logs/RecurrentPPO_Geometry_xxxxxx/final_model/vec_normalize.pkl \
    --method all \
    --seed 42
```

## 输出文件

评估结果会保存为 JSON 文件，包含：

- `baseline_zero_action_eval.json`: 零动作 baseline 结果
- `baseline_optimizer_eval.json`: ZF+SDR optimizer 结果
- `drl_model_results.json` / `final_trajectory_eval.json`: DRL 模型结果
- `comparison_summary.json`: 所有方法的汇总结果

每个结果文件包含：

- `steps`: 每个时间步的索引
- `P_BS`, `P_sat`, `P_total`: 功耗数据
- `SINR_UE`, `SINR_SUE`: SINR 数据 (dB)
- `SINR_min_dB`: 最小 SINR (dB)
- `success`: 是否满足约束 (True/False)
- `true_elevation`, `true_azimuth`: 卫星角度
- `summary`: 统计摘要 (均值、成功率等)

## 结果分析示例

训练结束时会打印对比摘要：

```
======================================================================
FINAL COMPARISON SUMMARY (DRL Model vs Baselines)
======================================================================
Method                     |   Avg P_total |   SINR_min |    Success
----------------------------------------------------------------------
Zero-Action Baseline       |        5.2341 |       8.45 |      30.0%
Baseline Optimizer        |        1.8234 |      11.23 |      95.0%
DRL Model (Final)        |        1.5432 |      11.56 |      98.0%
======================================================================
```

## 性能考虑

- **Zero-Action**: 最快，无计算开销
- **Baseline Optimizer**: 较慢，每个 step 需要 SDR 求解 (约 0.1-0.5s/step)
- **DRL Model**: 最快，只需要前向传播 (约 0.001-0.005s/step)

建议在调试阶段使用 `--skip-zero-baseline` 和不使用 `--run-baseline-optimizer` 来加快训练速度。
