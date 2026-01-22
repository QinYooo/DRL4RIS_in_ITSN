# Project Context: DRL-based RIS-Assisted ITSN Interference Mitigation

## 1. Project Background (Graduate Thesis Extension)
I am extending a published paper on **Energy-Efficient RIS-Assisted Co-channel Interference Mitigation in Integrated Terrestrial-Space Networks (ITSN)**.
* **Original Approach:** Used static optimization (BCD, SDR, SOCP) to minimize power consumption.
* **Current Problem:**
    1.  **Dynamic Environment:** LEO satellites move fast, causing CSI aging. Static algorithms are too slow.
    2.  **Discrete Constraints:** RIS hardware has limited resolution (e.g., 1-bit or 2-bit), making convex optimization difficult and computationally expensive.
* **My Innovation (The Solution):**
    I am proposing a **"Deep Reinforcement Learning (DRL) Assisted Hybrid Optimization Framework"** with **Ephemeris-Awareness**.
    * **DRL Agent (PPO/SAC):** Handles the difficult discrete RIS phase shifts and long-term adaptation to satellite motion.
    * **Mathematical Model (Model-based):** Handles the BS active beamforming (using Zero-Forcing or closed-form solutions) instantly inside the environment step, as this is convex and fast.
    * **Robustness:** The state space includes satellite ephemeris information to handle motion and potential errors.

## 2. Mathematical Framework & DRL Design

### A. State Space (Ephemeris-Aware)
The state $s_t$ must capture the dynamic geometry without suffering from the curse of dimensionality.
$$s_t = \{ \hat{\theta}_t, \hat{\phi}_t, \Delta\hat{\theta}_t, \Delta\hat{\phi}_t, \xi_{orbit}, t_{norm}, \mathbf{h}_{local} \}$$
* $\hat{\theta}_t, \hat{\phi}_t$: Satellite Elevation/Azimuth (from Ephemeris, with potential noise added during training for robustness).
* $\Delta\hat{\theta}_t, \Delta\hat{\phi}_t$: Angular velocity (trend).
* $t_{norm}$: Normalized time step within the pass.
* $\mathbf{h}_{local}$: Local CSI at BS (assumed known).

### B. Action Space
* **Action:** RIS Phase Shifts $\Phi_t$.
* **Constraint:** Discrete phases (quantized). The network output should be mapped to the nearest discrete phase (e.g., 0, $\pi$ for 1-bit).

### C. Hybrid Execution (Inside the Env Step)
For every step $t$:
1.  Agent outputs RIS phase $\Phi_t$.
2.  **Env calculates BS Beamforming ($W_t$)**: Based on fixed $\Phi_t$, calculate $W_t$ using **Zero-Forcing (ZF)** or Maximum Ratio Transmission (MRT) to satisfy user constraints instantly.
3.  **Env calculates Power ($P_{total}$)**: Sum of BS power and Satellite power.

### D. Reward Function
$$r_t = - (P_{BS, t} + P_{Sat, t}) - \lambda \cdot \text{Penalty}(SINR < \text{Threshold})$$
The goal is to minimize power while satisfying SINR constraints.

## 3. Existing Codebase (Scenario Generation)
I have already ported the MATLAB scenario generation to Python in **scenario.py**. This class handles channel generation, satellite movement, and geometry. **Use this class as the backbone for the Gym Environment.**

## 4. Coding Tasks Requirements
Please proceed to implement the following using Python and PyTorch (preferably using stable-baselines3 for RL or a clean custom PPO implementation):
1. Gym Environment Implementation (ITSNEnv):
- Wrap the provided ITSNScenario class into a gym.Env.
- Reset: Initialize a satellite pass (random trajectory logic).
- Step:
- - Accept action (Discrete RIS phases).
- - Compute equivalent channel $H_{eq}$.
- - CRITICAL: Implement Zero-Forcing (ZF) calculation for BS beamforming $W$ inside the step function to minimize power while satisfying rate constraints.Compute SINR and Power.
- - Return state, reward, done, info.
- State Space: Implement the "Ephemeris-Aware" state vector mentioned in Section 2A.
2. Ephemeris Error Simulation:
- In the environment, add a mechanism to inject noise into the observed satellite angle (State) vs. the real satellite angle used for channel generation. This trains the agent to be robust.
3. DRL Agent:
- Set up a PPO agent configuration suitable for this continuous/discrete hybrid nature (if using discrete actions for RIS, handle the dimension issues, or use continuous output + quantization layer).
4. Training Loop:
- Provide a script to train the agent over multiple satellite passes.Please generate the code in a modular way.

## 5. Debug Guidelines (Token-Efficient Mode)

### Debug 流程（省 Token）
- **根因假设**：用 1-3 句总结最可能的根因，不展开推理过程。
- **最小验证步骤**：给出下一步最小验证步骤（最多 3 条）。
- **按需详解**：只在明确要求时才给详细解释、背景或替代方案。

### 代码改动原则
- **最小 Diff**：优先局部修复，不重构，不改风格。
- **单次改动**：每次只做一个可验证的改动，然后说明如何验证。
- **要点输出**：尽量用要点列表，避免长段落。

### 阅读代码（省 Token）
- **不复述代码**：不要复述/粘贴已有的代码。
- **只引用关键行**：只引用要改的那几行（最多 15 行），并解释"为什么这几行"。
- **按需上下文**：若需要更多上下文，先让用户提供具体文件/函数，而不是猜测。

### Debug 以日志为中心
- **关键信号**：先指出日志里最关键的 1-2 行信号（错误码/堆栈顶/关键变量）。
- **假设精简**：给出一个最可能假设 + 一个备选假设即可。
- **不展开推理**：不要展开长推理。

### 执行命令（省 Token）
- **单条命令**：每次最多给 1 条命令（除非必须串联），并说明预期输出是什么。
- **分步执行**：如果需要多步，先给步骤 1，等用户贴输出再继续。

### 确定修复方案后
- **输出 Diff**：只输出 git diff（unified diff）。
- **精简说明**：diff 外最多 3 条 bullet：改了什么、为什么、怎么验证。
- **不解释细节**：不要额外解释实现细节。

### 验证策略（省 Token）
- **最短验证**：先给一个最短验证（例如运行单个测试/复现命令）。
- **渐进深入**：如果失败，再给第二个更深入验证。
- **不列测试矩阵**：不要一次性列出一堆测试矩阵。

### 测试脚本创建原则
- **需用户同意**：在创建任何测试/验证脚本之前，必须先征得用户同意。
- **不主动创建**：完成代码修改后，不要主动创建测试脚本，除非用户明确要求。
- **简要说明**：可以简要说明如何验证（1-2 句），但不要直接创建脚本文件。