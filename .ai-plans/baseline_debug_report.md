# Baseline Optimizer Debug Report

## 问题总结

### 1. SDR求解器卡死 (已解决)
**根因**：
- SDR问题规模太大（101x101矩阵，100个RIS元素）
- SCS和CVXOPT求解器都无法在合理时间内求解

**解决方案**：
- 添加了`ris_method`参数，支持'sdr'和'simple'两种方法
- 实现了`SimpleRISOptimizer`类，使用随机搜索优化RIS相位
- 默认使用'simple'方法，速度快但可能次优

**代码修改**：
- `baseline_optimizer.py`: 添加`_optimize_ris_simple()`方法
- `simple_ris_optimizer.py`: 新文件，实现随机搜索和坐标下降

### 2. 卫星功率爆炸 (已解决)
**根因**：
- `_update_satellite_power()`没有上限约束
- 当卫星SINR极低时，`P_s = gamma_j * interference / signal`会爆炸

**解决方案**：
- 添加`P_sat_max = 100W`上限
- 修改更新公式：`self.P_s = min(P_s_required, self.P_sat_max / self.N_s)`

**代码修改**：
```python
# baseline_optimizer.py line 52
self.P_sat_max = 100.0  # Maximum satellite power (W)

# baseline_optimizer.py line 735
self.P_s = min(P_s_required, self.P_sat_max / self.N_s)
```

### 3. 卫星SINR极低 (未完全解决)
**现象**：
- 卫星SINR约-110 dB，远低于10 dB目标
- BS对卫星的干扰过强

**可能原因**：
1. RIS相位优化不够好（随机搜索样本数太少）
2. ZF波束成形只考虑了BS用户，没有考虑对卫星的干扰
3. 信道条件差，BS到卫星的直射路径很强

**建议改进**：
1. 增加随机搜索样本数（当前50，可增加到200-500）
2. 使用坐标下降代替随机搜索
3. 修改ZF约束，同时考虑BS和卫星用户
4. 调整场景参数（RIS位置、卫星角度等）

## 当前性能

### 测试结果（3次迭代）
```
Iteration 1: Sum Rate = 16.14 bps/Hz, P_sum = 1423.74 W
Iteration 2: Sum Rate = 11.59 bps/Hz, P_sum = 1542.90 W
Iteration 3: Sum Rate = 12.21 bps/Hz, P_sum = ~1500 W
```

### SINR表现
- BS用户：大部分满足10 dB目标（部分略低）
- 卫星用户：-110 dB（严重不满足）

## 使用方法

### 简化版baseline（推荐）
```python
optimizer = BaselineZFSDROptimizer(
    K=K, J=J, N_t=N_t, N_s=N_s, N=N_ris,
    P_max=P_bs_max,
    sigma2=P_noise,
    gamma_k=gamma_k,
    gamma_j=gamma_j,
    P_b=P_bs_scale,
    N_iter=5,
    ris_method='simple',  # 使用简化方法
    verbose=True
)
```

### SDR版baseline（不推荐，会卡死）
```python
optimizer = BaselineZFSDROptimizer(
    ...,
    ris_method='sdr',  # 会卡死
    verbose=True
)
```

## 下一步工作

### 短期（必须）
1. 增加随机搜索样本数到200
2. 测试坐标下降方法
3. 验证多个场景下的性能

### 中期（建议）
1. 实现更好的RIS优化算法（如交替优化）
2. 修改ZF约束同时考虑卫星用户
3. 添加功率控制策略

### 长期（可选）
1. 尝试其他SDR求解器（MOSEK商业版）
2. 实现分布式优化算法
3. 添加信道估计误差鲁棒性

## 文件清单

### 修改的文件
- `baseline/baseline_optimizer.py`: 主优化器
  - 添加`ris_method`参数
  - 添加`_optimize_ris_simple()`方法
  - 修复卫星功率爆炸问题

### 新增的文件
- `baseline/simple_ris_optimizer.py`: 简化RIS优化器
- `baseline/test_debug.py`: 调试测试脚本
- `baseline/test_simple_baseline.py`: 简化版测试脚本
- `baseline/test_sdr_solver.py`: SDR求解器测试

## 验证命令

```bash
# 测试简化版baseline
cd baseline && python test_simple_baseline.py

# 查看调试信息
cd baseline && python test_debug.py
```
