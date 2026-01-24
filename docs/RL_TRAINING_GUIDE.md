# Training RL Agent with Autoencoder-Compressed State

## 快速开始

### 1. 确认AE模型已训练
```bash
# 检查checkpoint是否存在
ls checkpoints/channel_ae/channel_ae_best.pth
```

如果不存在，先训练AE：
```bash
python scripts/train_channel_ae.py
```

### 2. 训练RL Agent
```bash
# 基础训练 (使用默认参数)
python train_rl_with_ae.py

# 自定义参数
python train_rl_with_ae.py \
    --ae-checkpoint checkpoints/channel_ae/channel_ae_best.pth \
    --total-timesteps 500000 \
    --n-envs 4 \
    --learning-rate 3e-4 \
    --max-steps 40 \
    --n-substeps 5 \
    --device cuda \
    --seed 42
```

**训练输出**:
- `logs/PPO_AE_YYYYMMDD_HHMMSS/` - 训练日志目录
  - `checkpoints/` - 定期保存的模型
  - `best_model/` - 最佳模型
  - `tensorboard/` - TensorBoard日志
  - `eval/` - 评估结果

### 3. 监控训练
```bash
# 启动TensorBoard
tensorboard --logdir logs/PPO_AE_YYYYMMDD_HHMMSS/tensorboard

# 浏览器打开 http://localhost:6006
```

### 4. 评估训练好的模型
```bash
python evaluate_rl_with_ae.py \
    --model-path logs/PPO_AE_YYYYMMDD_HHMMSS/best_model/best_model.zip \
    --ae-checkpoint checkpoints/channel_ae/channel_ae_best.pth \
    --n-episodes 100 \
    --deterministic
```

**评估输出**:
- `results/evaluation_results.npz` - 评估数据
- `results/evaluation_plots.png` - 可视化结果

---

## 参数说明

### 训练参数 (train_rl_with_ae.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ae-checkpoint` | `checkpoints/channel_ae/channel_ae_best.pth` | AE模型路径 |
| `--total-timesteps` | 500000 | 总训练步数 |
| `--n-envs` | 4 | 并行环境数量 |
| `--learning-rate` | 3e-4 | 学习率 |
| `--max-steps` | 40 | 每个episode最大步数 |
| `--n-substeps` | 5 | 每个RL步的物理子步数 |
| `--phase-bits` | 4 | RIS相位量化位数 |
| `--latent-dim` | 32 | AE潜在空间维度 |
| `--device` | cuda | 设备 (cuda/cpu) |
| `--seed` | 42 | 随机种子 |
| `--checkpoint-freq` | 50000 | checkpoint保存频率 |

### 评估参数 (evaluate_rl_with_ae.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | (必需) | 训练好的模型路径 |
| `--ae-checkpoint` | `checkpoints/channel_ae/channel_ae_best.pth` | AE模型路径 |
| `--n-episodes` | 100 | 评估episode数量 |
| `--deterministic` | False | 是否使用确定性动作 |
| `--output-dir` | results | 结果输出目录 |

---

## 评估指标

评估脚本会输出以下指标：

### 1. Reward
- 平均奖励
- 标准差
- 最小/最大值

### 2. Success Rate
- 满足SINR约束的比例
- 平均成功率

### 3. Power Consumption
- BS功率 (W)
- 卫星功率 (W)
- 总功率 (W)

### 4. Episode Length
- 平均episode长度

---

## 可视化

评估脚本生成4个图表：

1. **Episode Reward Distribution** - 奖励分布直方图
2. **Success Rate Distribution** - 成功率分布
3. **Power Consumption (BS vs Satellite)** - 功率散点图
4. **Total Power Distribution** - 总功率分布

---

## 示例工作流

```bash
# 1. 训练AE (如果还没有)
python scripts/train_channel_ae.py

# 2. 训练RL agent
python train_rl_with_ae.py --total-timesteps 500000 --n-envs 4

# 3. 监控训练 (另一个终端)
tensorboard --logdir logs

# 4. 训练完成后评估
python evaluate_rl_with_ae.py \
    --model-path logs/PPO_AE_20240115_123456/best_model/best_model.zip \
    --n-episodes 100 \
    --deterministic

# 5. 查看结果
# - results/evaluation_plots.png
# - results/evaluation_results.npz
```

---

## 性能对比

要对比手工特征 vs AE特征的性能：

### 1. 训练baseline (手工特征)
```python
from envs.itsn_env import ITSNEnv
from stable_baselines3 import PPO

env = ITSNEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=500000)
model.save('logs/baseline_manual_features')
```

### 2. 训练AE版本
```bash
python train_rl_with_ae.py --total-timesteps 500000
```

### 3. 对比评估结果
- 收敛速度 (TensorBoard曲线)
- 最终性能 (功耗、成功率)
- 计算开销 (训练时间)

---

## 故障排除

### 问题1: CUDA out of memory
**解决**: 减少并行环境数量
```bash
python train_rl_with_ae.py --n-envs 2
```

### 问题2: 训练不稳定
**解决**: 调整学习率或clip range
```bash
python train_rl_with_ae.py --learning-rate 1e-4 --clip-range 0.1
```

### 问题3: AE checkpoint not found
**解决**: 先训练AE
```bash
python scripts/train_channel_ae.py
```

---

## 高级用法

### 超参数调优
使用Optuna进行超参数搜索：
```python
import optuna
from train_rl_with_ae import train_rl_agent

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-3, 0.1)

    model, log_path = train_rl_agent(
        ae_checkpoint_path='checkpoints/channel_ae/channel_ae_best.pth',
        learning_rate=lr,
        ent_coef=ent_coef,
        total_timesteps=100000
    )

    # Return evaluation metric
    return evaluate_model(model)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 分布式训练
使用多GPU训练：
```bash
# 在不同GPU上启动多个训练
CUDA_VISIBLE_DEVICES=0 python train_rl_with_ae.py --seed 42 &
CUDA_VISIBLE_DEVICES=1 python train_rl_with_ae.py --seed 43 &
```

---

## 预期结果

基于初步实验，预期性能：

| 指标 | 手工特征 | AE特征 |
|------|----------|--------|
| 收敛步数 | ~300k | ~250k (更快) |
| 最终功耗 | ~X.XX W | ~X.XX W (待测) |
| 成功率 | ~XX% | ~XX% (待测) |
| 训练时间 | ~X小时 | ~X小时 (+15-20%) |

**注**: 具体数值需要实际训练后更新。
