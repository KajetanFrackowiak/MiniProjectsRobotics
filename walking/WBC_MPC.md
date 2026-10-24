Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control (Kim et al., 2019)

# Source code:
Not specified
# Webpage/Videos
https://youtu.be/6JlVol3eyNI

# What and where is the current problem?
- Current WBC methods focus on how to follow given contact trajectory omitting flight trajectories which is needed mainly during running
- Since WBC can only predict one step ahead they consider MPC integration, but current WBC + MPC approaches are not efficient enough, they mention that some approaches for example use fully-body MPC which gives MPC much more to compute

# How the authors try to solve the problem?
```
In this paper, we propose a controller combining whole-body control (WBC) and model predictive control (MPC). In our framework, MPC finds an optimal reaction force profile over a longer time horizon with a simple model, and WBC computes joint torque, position, and velocity commands based on the reaction forces computed from MPC. Unlike existing WBCs, which attempt to track commanded body trajectories, our controller is focused more on the reaction force command, which allows it to accomplish high speed dynamic locomotion with aerial phases.
```
Their MPC works 40 Hz = 0.04 kHz & WBIC on 500 Hz = 0.5 kHz
![[Pasted image 20260827211104.png|521]]

![[Pasted image 20260827210522.png]]

They used Kalman Filter-based state estimator to compute global body position and velocity based on kinematics and acceleration data.
## Model Predictive Control
### 1. MPC Formulation
>Key: They don't solve the full nonlinear robot dynamics online. They freeze/approximate the nonlinear parts around the desired trajectory, turning the force-selection problem into a convex QP.

#### What does MPC do?
The MPC predicts the robot's future states over a horizon:

$$x_{k+1}, x_{k+2}, \ldots, x_{k+N}$$
by choosing future contact forces:

$$f_k, f_{k+1}, \ldots, f_{k+N-1}$$
##### State update equation:
$$x_{k+1} = \begin{bmatrix} \Theta_{k+1} \\ p_{k+1} \\ \omega_{k+1} \\ \dot{p}_{k+1} \end{bmatrix}$$
##### Position:
$$p_{k+1} = p_k + \dot{p}_k \Delta t$$
**Meaning:** $\text{new position} = \text{old position} + \text{velocity} \times \text{time}$.

##### Linear Velocity:
$$\dot{p}_{k+1} = \dot{p}_k + \frac{\Delta t}{m}\sum_i f_i + g\Delta t$$

**Meaning:** $\text{new velocity} = \text{old velocity} + \text{acceleration} \times \text{time}$

##### Angular Velocity
$$\omega_{k+1} = \omega_k + I^{-1}\sum_i [r_i]_\times f_i \Delta t$$
**Meaning:** $\text{new angular velocity} = \text{old angular velocity} + \text{angular acceleration} \times \text{time}$.

##### Orientation
$$\Theta_{k+1} = \Theta_k + R_z(\psi_k)\omega_k \Delta t$$

**Meaning:** $\text{new orientation} = \text{old orientation} + \text{angular velocity} \times \text{time}$, using the appropriate frame transformation.

#### Optimization Problem
The MPC solves the following constrained minimization:
$$\min_{\{x_k, f_k\}} \sum_{k=0}^{N-1} \left( \Vert{}x_{k+1} - x_k^{\text{ref}}\Vert{}_Q^2 + \Vert{}f_k\Vert{}_R^2 \right)$$
**Subject to:**
- **System Dynamics:**
    $$x_{k+1} = A_k x_k + B_k f_k + \hat{g}$$
- **Friction Cone Constraints:**
    $$\vert{}f_x\vert{} \le \mu f_z$$
$$\vert{}f_y\vert{} \le \mu f_z$$$$f_z > 0$$Therefore, because:
- dynamics are **linear**
- objective is **quadratic**
- constraints are **linear**
the MPC optimization is a **Quadratic Program (QP)**.

### 2. Gait Scheduler
They use a periodic phase-based gait scheduler introduced in [[Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains]] that needs only two parameters to specify a gait type; phase offset and stance period for each foot.

### 3. Footstep Planner
They choose the upcoming footstep location with the following equation:

$$\underbrace{r_i^{\text{cmd}}}_{\text{Target Step}} = \underbrace{p_{\text{shoulder}, i}}_{\text{Nominal Base}} + \underbrace{p_{\text{symmetry}}}_{\text{Linear Motion}} + \underbrace{p_{\text{centrifugal}}}_{\text{Turning Motion}}$$

Where:
- $r_i^{\text{cmd}}$:  Target Footstep Location - The commanded 3D position where the $i$-th foot should touch down in the global frame.

- $p_{\text{shoulder}, i} = p_k + R_z(\psi_k) l_i$
	- **Role:** Neutral Shoulder Reference.
	- **Meaning:** Computes where the hip/shoulder joint sits in the global coordinate frame right now.

- $p_{\text{symmetry}} = \frac{t_{\text{stance}}}{2} v + k(v - v_{\text{cmd}})$
	- **Role:** Raibert Heuristic (Forward/Lateral Balance).
	- **Meaning:**
	    - $\frac{t_{\text{stance}}}{2} v$: Places the foot directly in the middle of the stance phase trajectory to balance acceleration and deceleration forces equally across contact.
	    - $k(v - v_{\text{cmd}})$: A proportional feedback offset that steps further out if the robot is drifting slower or faster than intended.

- $p_{\text{centrifugal}} = \frac{1}{2} \sqrt{\frac{h}{g}} (v \times \omega_{\text{cmd}})$
	- **Role:** Turning Offset (Inverted Pendulum Banking).
	- **Meaning:** Computes a side-step offset during turns to lean into the corner, counteracting the centrifugal force generated by combining forward speed $v$ with rotational velocity $\omega_{\text{cmd}}$. The factor $\sqrt{\frac{h}{g}}$ is the natural time constant of the simplified Linear Inverted Pendulum Model (LIPM).

## Whole-Body Impulse Control
> Using the reaction forces found by the MPC, the WBIC computes joint position, velocity, and torque commands.

### 1. Prioritized Task Execution
WBIC has multiple tasks with a strict priority hierarchy, e.g.:
$$
\text{Body Orientation}
>
\text{Body Position}
>
\text{Foot Position}
$$
Each task has a Jacobian:

$$
J_i
$$
relating joint motion to task motion:
$$
\dot{x}_i = J_i\dot{q}
$$
#### Task Position

The position error is:

$$
e_i=x_i^{des}-x_i
$$

The joint displacement is computed recursively:

$$
\boxed{
\Delta q_i
=
\Delta q_{i-1}
+
J_{i|pre}^{\dagger}
\left(
e_i-J_i\Delta q_{i-1}
\right)
}
$$

**Meaning:** solve task \(i\) using the remaining degrees of freedom after satisfying all higher-priority tasks.

The projected Jacobian is:
$$
J_{i|pre}=J_iN_{i-1}
$$
where $N_{i-1}$ projects motion into the null space of previous tasks.

#### Joint Position Command
After obtaining the joint displacement:
$$
\boxed{
q_j^{cmd}=q_j+\Delta q_j
}
$$
**Meaning:** desired joint position = current joint position + required joint displacement.
This is sent to the joint-level PD controller.

--- 
### 2. Desired Joint Velocity
The desired joint velocity is computed using prioritized inverse kinematics:

$$
\boxed{
\dot{q}_i^{cmd}
=
\dot{q}_{i-1}^{cmd}
+
J_{i|pre}^{\dagger}
\left(
\dot{x}_i^{des}
-
J_i\dot{q}_{i-1}^{cmd}
\right)
}
$$

with:

$$
\dot{q}_0^{cmd}=0
$$

**Meaning:** generate the joint velocity needed for task \(i\), while preserving all higher-priority tasks.

---
### 3. Desired Joint Acceleration
For acceleration control, WBIC uses a **dynamically consistent pseudoinverse** (it takes robot mass/inertia matrix A into account):

$$
\boxed{
J^{\#}
=
A^{-1}J^T
\left(
JA^{-1}J^T
\right)^{-1}
}
$$

The desired task acceleration is:

$$
\boxed{
\ddot{x}_i^{cmd}
=
\ddot{x}_i^{des}
+
K_p(x_i^{des}-x_i)
+
K_d(\dot{x}_i^{des}-\dot{x}_i)
}
$$

**Meaning:** desired task acceleration = feedforward acceleration + position error correction + velocity error correction.

The prioritized joint acceleration is:

$$
\boxed{
\ddot{q}_i^{cmd}
=
\ddot{q}_{i-1}^{cmd}
+
J_{i|pre}^{\#}
\left(
\ddot{x}_i^{cmd}
-
\dot{J}_i\dot{q}
-
J_i\ddot{q}_{i-1}^{cmd}
\right)
}
$$

where the dynamically consistent projected Jacobian is used.

---

### 4. WBIC Quadratic Program

The acceleration commands from the previous step and the reaction forces from MPC are **not necessarily perfectly dynamically consistent** with the full robot.

Therefore, WBIC solves another QP.

#### Reaction Force Correction
Start with the MPC reaction force:

$$
f_r^{MPC}
$$
and allow a small correction:
$$
\boxed{
f_r=f_r^{MPC}+\delta f_r
}
$$

where $\delta f_r$ should ideally be small.

The desired acceleration can also be relaxed:

$$
\boxed{
\ddot{q}
=
\ddot{q}^{cmd}
+
\begin{bmatrix}
\delta f\\
0_{n_j}
\end{bmatrix}
}
$$

---
#### QP Objective

$$
\boxed{
\min_{\delta f_r,\delta f}
\quad
\delta f_r^TQ_1\delta f_r
+
\delta f^TQ_2\delta f
}
$$

**Meaning:**

* minimize deviation from the MPC reaction forces
* minimize deviation from the desired acceleration

---

#### Full-Body Dynamics Constraint

The full-body robot dynamics are:

$$
\boxed{
\begin{bmatrix}
\tau_f\\
\tau_j
\end{bmatrix}
=
A\ddot{q}+b+g-J_c^Tf_r
}
$$

where:
* $J_c$ - contact Jacobian
* $f_r$ - reaction forces

Because the floating base is unactuated:

$$
\tau_f=0
$$

Therefore, WBIC enforces the floating-base dynamics:

$$
\boxed{
S_f(A\ddot{q}+b+g)
=
S_fJ_c^Tf_r
}
$$

This makes the reaction forces dynamically consistent with the floating-base dynamics.

---

#### Contact Force Constraints

The reaction forces must satisfy:

$$
\boxed{
Wf_r\geq0
}
$$

which represents the contact-force inequality constraints, such as friction constraints.

---

### 5. Final Joint Torque
After the QP finds:
$$
f_r
$$
and
$$
\ddot{q}
$$
WBIC plugs them into the full-body dynamics:
$$
\boxed{
\begin{bmatrix}
\tau_f\\
\tau_j
\end{bmatrix}
=
A\ddot{q}+b+g-J_c^Tf_r
}
$$
and extracts the joint torque:

$$
\boxed{
\tau_j
}
$$
as the final torque command.

---

## MPC + WBIC

$$
\boxed{
\text{MPC}
\rightarrow
f^{MPC}
\rightarrow
\text{WBIC}
\rightarrow
\tau_j
}
$$

### MPC

> **What ground reaction forces should the robot produce over the next horizon?**

MPC uses a **simplified lumped-mass model** and provides the future-looking force plan.

### WBIC

> **Given those desired forces and my whole-body tasks, what should I command to the joints right now?**

WBIC uses the **full-body robot dynamics** to make the force plan physically consistent and convert it into joint commands.

### Key Idea

$$
\boxed{\text{MPC = look-ahead / force planning}}
$$
$$
\boxed{\text{WBIC = whole-body realization / force-to-torque control}}
$$



# Setup
The authors use Mini-Cheetah robot

**Task-Space Gains [[#3. Desired Joint Acceleration| WBC part 3]]:
- Body orientation $k_p = [100, 100, 100]^T$, $k_d = [10,10,10]^T$ 
- Body position $k_p = [100, 100, 100]^T$, $k_d = [10,10,10]^T$
- Foot position $k_p = [100, 100, 100]^T$, $k_d = [10,10,10]^T$

**Low-Level Joint PD Gains (In low-level joint actuators)**
- All feedback controller $k_p =3 Nm/rad$
- All feedback controller $k_d = 3Nms/rad$ except **abduction gain**, which have $k_d = 1 Nm/s$  

**Weights in QP problem**:
- Reaction force $Q_1 = 1$
- Floating base control $Q_2 = 0.1$


# Results
![[Pasted image 20260827212924.png]]
(FL) means Front left leg there

The results on figure 4e showed that their WBIC very well was following MPC. Figure 4a shows that the robot is able to not hook on own legs.

Authors writes that the maximum stable running record on the treadmill was 3.7 m/s. 

Authors also writes the the robot was able to go in various gait types and terrains (wet grass , or rough terrain like on little stones) outside. They writes that the robot was able to run at speeds over 1 m/s with every gait type.

