Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains (Bledt et al., 2018)

# Source code:
Not specified
# Webpage/Videos
Not specified

# What and where is the current problem?
Authors write that current force estimation  by control systems in uncertain terrain is not enough, since it is too noisy in such terrains and notoriously fragile.

# How the authors try to solve the problem?
```
This paper introduces an approach to probabilistically fuse contact models, managing uncertainty in terrain geometry, dynamic modeling, and kinematics to improve the robustness of contact initiation at touchdown
```

### 1. Discrete-Time Generalized Momentum Observer
Standard generalized momentum observers integrate continuous-time dynamics. To eliminate fictitious disturbance torques during discrete implementation, the authors apply summation by parts over the Z-domain filter:

$$\frac{1-\gamma}{1-\gamma z^{-1}}\tau_d = \frac{1-\gamma}{1-\gamma z^{-1}}\left(M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) - S^\top \tau\right)$$


Applying summation by parts yields the discrete-time feedforward disturbance torque observer $\hat{\tau}_d$:

$$\hat{\tau}_d = \beta p_k - \frac{1-\gamma}{1-\gamma z^{-1}}\left(\beta p + S^\top \tau + C^\top(q,\dot{q})\dot{q} - g(q)\right)$$

where the parameters and state vectors are defined as:
- $p = M(q)\dot{q}$ is the generalized momentum.
- $\gamma = e^{-\lambda \Delta t}$ maps the continuous cutoff frequency $\lambda$ to the Z-domain pole for sample period $\Delta t$.
- $\beta = \frac{(1-\gamma)\gamma^{-1}}{\Delta t}$ scales the momentum feedforward term.

The foot disturbance force estimate $\hat{f}_i$ for leg $i$ is extracted using the leg Jacobian $J_i$ and selector matrix $S_{l_i}$:
$$\hat{f}_i = \left(S_{l_i} J_i^\top\right)^{-1} S_{l_i} \hat{\tau}_d$$
### 2. Probabilistic Prior Models
#### Prediction Model (Gait Phase Prior)
The prediction prior models contact probability $P(c \mid s_\phi, \phi)$ based on scheduled state $s_\phi \in \{0=\text{swing}, 1=\text{contact}\}$ and subphase $\phi \in [0, 1)$:


$$P(c \mid s_\phi, \phi) = \frac{1}{2}\left( s_\phi \left[\operatorname{erf}\left(\frac{\phi-\mu_{c_0}}{\sigma_{c_0}\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\mu_{c_1}-\phi}{\sigma_{c_1}\sqrt{2}}\right)\right] + \bar{s}_\phi \left[2 + \operatorname{erf}\left(\frac{\mu_{\bar{c}_0}-\phi}{\sigma_{\bar{c}_0}\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\phi-\mu_{\bar{c}_1}}{\sigma_{\bar{c}_1}\sqrt{2}}\right)\right] \right)$$

#### Measurement Model 1 (Foot Height Prior)
Using vertical foot position $p_z$ and assumed ground height distribution $Z_g \sim \mathcal{N}(\mu_{z_g}, \sigma_{z_g}^2)$:

$$P(c \mid p_z) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{\mu_{z_g} - p_z}{\sigma_{z_g}\sqrt{2}}\right)\right]$$
#### Measurement Model 2 (Estimated Force Prior)
Using estimated vertical foot force $f_z$ and expected contact initiation force distribution $F_c \sim \mathcal{N}(\mu_{f_c}, \sigma_{f_c}^2)$:

$$P(c \mid f_z) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{f_z - \mu_{f_c}}{\sigma_{f_c}\sqrt{2}}\right)\right]$$

### 3. Kalman Filter Probabilistic Fusion
The state vector $\hat{x}_k \in \mathbb{R}^N$ tracks the continuous probability of contact across $N$ feet ($N=4$):

$$\hat{x}_k = \begin{bmatrix} P_1(c) \\ \vdots \\ P_N(c) \end{bmatrix}_k$$
#### Prediction Step
The system matrices are set to $A_k = 0_N$ and $B_k = I_N$, turning the prediction into an instantaneous prior update driven by the gait phase input $u_k$:

$$\hat{x}_{k \mid k-1} = u_k = \begin{bmatrix} P_1(c \mid s_\phi, \phi) \\ \vdots \\ P_N(c \mid s_\phi, \phi) \end{bmatrix}_k, \quad \Sigma_{k \mid k-1} = \Sigma_{w_k}$$

#### Correction Step
The observations are stacked into $\tilde{z}_k \in \mathbb{R}^{2N}$ with output matrix $H_k = \begin{bmatrix} I_N \\ I_N \end{bmatrix}$:

$$\tilde{z}_k = \begin{bmatrix} \tilde{z}_{1,k} \\ \tilde{z}_{2,k} \end{bmatrix} = \begin{bmatrix} P_1(c \mid p_z) & \dots & P_N(c \mid p_z) \\ P_1(c \mid f_z) & \dots & P_N(c \mid f_z) \end{bmatrix}_k^\top$$

$$K_k = \Sigma_{k \mid k-1} H_k^\top \left(H_k \Sigma_{k \mid k-1} H_k^\top + \Sigma_{v_k}\right)^{-1}$$
$$\hat{x}_{k \mid k} = \hat{x}_{k \mid k-1} + K_k \left(\tilde{z}_k - H_k \hat{x}_{k \mid k-1}\right)$$

### 4. Classification & Optimization Objective
#### Binary Decision Rule
The binary state $\hat{s}_i \in \{0, 1\}$ for each foot $i$ is obtained by thresholding the posterior probability:

$$\hat{s}_i = \begin{cases} 1, & \text{if } P_i(c) \ge P_c(c) \\ 0, & \text{if } P_i(c) < P_c(c) \end{cases}$$

#### Parameter Optimization Objective
The covariance hyper-parameters $\chi$ for the probabilistic models are fitted offline by minimizing:
$$\chi^* = \arg\min_\chi \left( (s - \hat{s})^\top W_c (s - \hat{s}) + (s - P(c))^\top W_n (s - P(c)) \right)$$


where $s$ is ground truth contact, $\hat{s}$ is the estimated binary state, and $W_c \gg W_n$ prioritizes classification accuracy.

### 5. Event-Based Gait Switching
![[Pasted image 20260829180027.png]]

To handle unexpected terrain variations (e.g., obstacles or depressions), the authors replace rigid, open-loop time-based switching with an **Event-Based Finite State Machine (FSM)** driven by the estimated contact state $\hat{s}$.

#### Control Paradigm Comparison
- **Time-Based Switching:** The leg controller relies solely on the open-loop scheduled gait state $s_\phi \in \{0=\text{swing}, 1=\text{stance}\}$ provided by the time-based Gait Scheduler.
    
- **Event-Based Switching:** The leg controller fuses the scheduled state $s_\phi$ with the detector output $\hat{s} \in \{0, 1\}$ within an Event-Based FSM to select appropriate low-level control laws.
    

#### Contact State FSM Modes
The FSM evaluates four distinct operational states for each leg independently:

1. **State: Swing ($\hat{s} = 0, s_\phi = 0$) — Nominal Swing**
    - **Condition:** Scheduled for swing ($s_\phi = 0$) and no contact detected ($\hat{s} = 0$).
        
    - **Control Action:** Tracks the nominal continuous swing foot trajectory (e.g., Bézier curve) using swing impedance control.
        
2. **State: Contact ($\hat{s} = 1, s_\phi = 1$) — Nominal Stance**
    - **Condition:** Scheduled for stance ($s_\phi = 1$) and contact confirmed ($\hat{s} = 1$).
        
    - **Control Action:** Actively engages the balance controller to produce desired ground reaction forces.
        
3. **State: Early ($\hat{s} = 1, s_\phi = 0$) — Early Touchdown (Obstacle / Elevated Ground)**
    - **Condition:** Scheduled for swing ($s_\phi = 0$), but contact detected prematurely ($\hat{s} = 1$).
        
    - **Control Action:**
        - Immediately aborts swing trajectory tracking to prevent collision forces.
            
        - Holds the current Cartesian foot position.
            
        - **Delay Filter ($t > t_0 + t_{\text{delay}}$):** Pauses before notifying the balance controller to prevent transient brushes or slips from corrupting body balance. Once persistent contact is confirmed beyond $t_{\text{delay}}$, the foot is made available for balance control.
            
4. **State: Late ($\hat{s} = 0, s_\phi = 1$) — Late Touchdown (Depression / Lower Ground)**
    - **Condition:** Scheduled for stance ($s_\phi = 1$), but no contact detected ($\hat{s} = 0$).
        
    - **Control Action:**
        
        - Inhibits force generation from the balance controller to prevent instigating body collapse.
            
        - Continues extending the leg downward to search for ground contact until $\hat{s} = 1$ is established.

# Setup
The authors used robot **Cheetah 3**.

### Table I: Probabilistic Gaussian Parameters

|**Category**|**Parameter**|**Mean (μ)**|**Units**|**Variance (σ2)**|**Units**|
|---|---|---|---|---|---|
|**Motion Model**|$\bar{c}_0$|$0$|—|$0.05$|—|
||$\bar{c}_1$|$1$|—|$0.05$|—|
||$c_0$|$0$|—|$0.05$|—|
||$c_1$|$1$|—|$0.05$|—|
|**Measurement Model**|$z_g$|$0$|$\text{m}$|$0.1$|$\text{m}^2$|
||$f_c$|$40$|$\text{N}$|$25$|$\text{N}^2$|
# Results
![[Pasted image 20260829180945.png]]

![[Pasted image 20260829181016.png]]

![[Pasted image 20260829181026.png]]

The results on fig 8, and fig 9 show that the Kalman filter very reliably predicts the contact state even in unexpected foot contact. 

Fig 11 clearly shows that using the event based switching (scheduled state $s_\phi$ with the detector output $\hat{s} \in \{0, 1\}$)  is really helpful mainly to unexpected events