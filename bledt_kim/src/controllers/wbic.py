import numpy as np

from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import LeafSystem
from pydrake.multibody.tree import JacobianWrtVariable


class WBICController(LeafSystem):
    """Whole-Body Impulse Controller (Kim et al., 2019).

    Current milestone: STANDING (all four feet in permanent stance).
    Task hierarchy with strict priority (paper, WBIC part 1-3):
        1. body orientation
        2. body position
    Tasks produce qdd_cmd through prioritized, dynamically-consistent
    inverse dynamics. The WBIC QP (paper, part 4) then finds a reaction-force
    correction (delta_fr) and a base-acceleration relaxation (delta_f) that
    make the (currently fake) MPC forces consistent with the floating-base
    dynamics. Output: joint torques.

    Extension points for later milestones:
      - swing-foot task: foot_pos_des / foot_vel_des ports are already wired
        but not yet used (Milestone 2, together with the gait scheduler).
      - q_cmd / v_cmd prioritized-kinematics outputs feeding the joint PD
        (paper, part 1-2); currently output zeros and are unconnected.
    """

    def __init__(
        self,
        plant,
        robot_model,
        mu=0.6,
        q1_weight=1.0,      # Q1: reaction-force tracking weight (paper: 1)
        # Q2: base-acceleration relaxation weight (paper: 0.1). Raised to 100
        # because the fake MPC gives no feedback; task accelerations dominate.
        # Revert to ~0.1 when a real MPC is used.
        q2_weight=100.0,
        kp_body_rot=100.0,
        kd_body_rot=10.0,
        kp_body_pos=100.0,
        kd_body_pos=10.0,
        kp_foot=100.0,      # reserved for the swing-foot task (paper: 100)
        kd_foot=10.0,       # reserved for the swing-foot task (paper: 10)
        tau_max=17.0,
        fz_min=5.0,
        fz_max=180.0,
        max_delta_fr=25.0,
    ):
        super().__init__()

        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.robot_model = robot_model

        self.num_positions = plant.num_positions()   # 19 (7 floating base + 12 joints)
        self.num_velocities = plant.num_velocities()  # 18 (6 floating base + 12 joints)
        self.num_joints = 12
        self.mu = mu

        self.q1_weight = q1_weight
        self.q2_weight = q2_weight

        # Task-space PD gains
        self.kp_body_rot = kp_body_rot
        self.kd_body_rot = kd_body_rot
        self.kp_body_pos = kp_body_pos
        self.kd_body_pos = kd_body_pos
        self.kp_foot = kp_foot
        self.kd_foot = kd_foot

        self.tau_max = tau_max
        self.fz_min = fz_min
        self.fz_max = fz_max
        self.max_delta_fr = max_delta_fr

        # Inputs
        self.DeclareVectorInputPort("x_full", self.num_positions + self.num_velocities)
        # f_mpc: MPC reaction forces, 4 feet x 3
        self.DeclareVectorInputPort("f_mpc", 12)
        # s_phi: contact state (1 = stance, 0 = swing)
        self.DeclareVectorInputPort("s_phi", 4)
        # foot_pos_des / foot_vel_des: desired swing-foot trajectories (reserved)
        self.DeclareVectorInputPort("foot_pos_des", 12)
        self.DeclareVectorInputPort("foot_vel_des", 12)
        # body_pose_des: desired body pose [x, y, z, r, p, y]
        self.DeclareVectorInputPort("body_pose_des", 6)
        # cmd_vel: commanded body velocity
        self.DeclareVectorInputPort("cmd_vel", 3)

        # Outputs
        self.DeclareVectorOutputPort("torque", self.num_joints, self.CalcTorque)
        self.DeclareVectorOutputPort("q_cmd", self.num_joints, self.CalcQCmd)
        self.DeclareVectorOutputPort("v_cmd", self.num_joints, self.CalcVCmd)

        # Contact frames: toe spheres touch the ground (shank frames are higher).
        self.foot_frames = [
            plant.GetFrameByName("toe_fr"),
            plant.GetFrameByName("toe_fl"),
            plant.GetFrameByName("toe_hr"),
            plant.GetFrameByName("toe_hl"),
        ]
        self.body = plant.GetBodyByName("body")
        self.body_frame = self.body.body_frame()

    def _update_plant_state(self, context):
        x = self.get_input_port(0).Eval(context)
        q = x[: self.num_positions]
        v = x[self.num_positions :]
        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)
        return q, v

    def _body_task_data(self, body_pose_des, cmd_vel=None):
        """Jacobians, bias accelerations and PD commands of the body tasks.

        Returns a priority-ordered task list:
            [(J_orientation, Jdot_v_orientation, xdd_cmd_orientation),
             (J_position,    Jdot_v_position,    xdd_cmd_position)]
        Task twist ordering is [angular; linear] in the world frame, matching
        Drake's spatial velocity convention.
        """
        X_WB = self.plant.EvalBodyPoseInWorld(self.plant_context, self.body)
        V_WB = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.body)

        J_body = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.body_frame,
            np.zeros(3),
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        # Bias acceleration Jdot*v of the body twist (6)
        A_bias = self.plant.CalcBiasSpatialAcceleration(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.body_frame,
            np.zeros(3),
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        Jdv_body = np.concatenate([A_bias.rotational(), A_bias.translational()])

        p_b = X_WB.translation()
        R_b = X_WB.rotation()
        w_b = V_WB.rotational()
        v_b = V_WB.translational()

        if cmd_vel is not None:
            vx_cmd, vy_cmd, wz_cmd = float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2])
        else:
            vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0

        yaw = body_pose_des[5]
        c, s = np.cos(yaw), np.sin(yaw)
        v_world_des = np.array([c * vx_cmd - s * vy_cmd, s * vx_cmd + c * vy_cmd, 0.0])
        w_world_des = np.array([0.0, 0.0, wz_cmd])

        # Task 1: body orientation
        R_des = RotationMatrix(RollPitchYaw(body_pose_des[3:]))
        e_rot = RollPitchYaw(R_des.multiply(R_b.transpose())).vector()
        xdd_rot = self.kp_body_rot * e_rot + self.kd_body_rot * (w_world_des - w_b)

        # Task 2: body position
        # For height (z): PD position tracking
        ez = body_pose_des[2] - p_b[2]
        xdd_z = self.kp_body_pos * ez - self.kd_body_pos * v_b[2]

        # For horizontal (x, y): track commanded velocity when moving, or hold position when standing
        if np.hypot(vx_cmd, vy_cmd) > 1e-3:
            xdd_x = self.kd_body_pos * (v_world_des[0] - v_b[0])
            xdd_y = self.kd_body_pos * (v_world_des[1] - v_b[1])
        else:
            xdd_x = self.kp_body_pos * (body_pose_des[0] - p_b[0]) - self.kd_body_pos * v_b[0]
            xdd_y = self.kp_body_pos * (body_pose_des[1] - p_b[1]) - self.kd_body_pos * v_b[1]

        xdd_pos = np.array([xdd_x, xdd_y, xdd_z])

        return [
            (J_body[:3], Jdv_body[:3], xdd_rot),
            (J_body[3:], Jdv_body[3:], xdd_pos),
        ]

    def _foot_task_data(self, s_phi, foot_pos_des, foot_vel_des, v):
        """Swing-foot task (lowest priority, below body).

        Only swing legs (s_phi==0) are included. Stance legs are
        force-controlled via the QP and must NOT be position-controlled.
        Returns None if no swing legs, otherwise single stacked task
        (J_foot, Jdv_foot, xdd_foot) with 3*N_swing rows.
        """
        swing_idx = [i for i in range(4) if s_phi[i] < 0.5]
        if not swing_idx:
            return None

        J_list = []
        Jdv_list = []
        e_list = []
        edot_list = []

        for i in swing_idx:
            J_f = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kV,
                self.foot_frames[i],
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame(),
            )
            # Bias acceleration Jdot*v for this foot (3,)
            Jdv_f = self.plant.CalcBiasTranslationalAcceleration(
                self.plant_context,
                JacobianWrtVariable.kV,
                self.foot_frames[i],
                np.zeros((3, 1)),
                self.plant.world_frame(),
                self.plant.world_frame(),
            ).ravel()

            # Current foot position / velocity in world
            p_cur = self.plant.CalcPointsPositions(
                self.plant_context,
                self.foot_frames[i],
                np.zeros((3, 1)),
                self.plant.world_frame(),
            ).ravel()
            # v_cur = J_f @ v  (v is generalized velocity 18)
            v_cur = J_f @ v

            p_des = foot_pos_des[3 * i : 3 * i + 3]
            v_des = foot_vel_des[3 * i : 3 * i + 3]

            e = p_des - p_cur
            edot = v_des - v_cur

            J_list.append(J_f)
            Jdv_list.append(Jdv_f)
            e_list.append(e)
            edot_list.append(edot)

        J_foot = np.vstack(J_list)
        Jdv_foot = np.concatenate(Jdv_list)
        e_stack = np.concatenate(e_list)
        edot_stack = np.concatenate(edot_list)

        # PD on foot: xdd = kp*e + kd*edot (feedforward accel = 0)
        xdd_foot = self.kp_foot * e_stack + self.kd_foot * edot_stack

        return (J_foot, Jdv_foot, xdd_foot)

    def _prioritized_ik(self, tasks, Jc=None, damping=1e-6):
        """Prioritized kinematic IK (paper, part 1-2) with null-space projection.

        tasks: list of (J (m x nv), err (m,)) where err = e for position or
               xd for velocity. Uses Moore-Penrose pseudoinverse with
               small damping for singularity robustness.
        Jc: contact Jacobian (nc*3 x nv) for N0 = I - Jc† Jc
        Returns delta (nv,) or qdot (nv,) depending on input.
        """
        nv = self.num_velocities
        delta = np.zeros(nv)
        # N0 from contacts (highest priority, Kim IV.A (20) N0 = I - Jc*Jc)
        if Jc is not None and Jc.shape[0] > 0:
            JJt = Jc @ Jc.T + damping * np.eye(Jc.shape[0])
            try:
                inv = np.linalg.inv(JJt)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(JJt)
            Jc_pinv = Jc.T @ inv
            N = np.eye(nv) - Jc_pinv @ Jc
        else:
            N = np.eye(nv)
        for J, err in tasks:
            if J.shape[0] == 0:
                continue
            JN = J @ N
            m = JN.shape[0]
            # Damped pseudoinverse: JN^T (JN JN^T + lambda*I)^-1
            JJt = JN @ JN.T
            JJt += damping * np.eye(m)
            try:
                inv = np.linalg.inv(JJt)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(JJt)
            JN_pinv = JN.T @ inv
            residual = err - J @ delta
            delta = delta + JN_pinv @ residual
            N = N - JN_pinv @ JN
        return delta

    def _prioritized_qdd(self, tasks, A, Jc=None):
        """Prioritized inverse dynamics (paper, WBIC part 3).

        Recursion with the dynamically consistent pseudoinverse:
            qdd_i = qdd_{i-1} + J_{i|pre}# (xdd_i_cmd - Jdot_i v - J_i qdd_{i-1})
            N_i   = N_{i-1} - J_{i|pre}# J_{i|pre}
        Jc: contact Jacobian for N0 = I - Jc# Jc (dynamically consistent)
        """
        nv = self.num_velocities
        A_inv = np.linalg.inv(A)
        qdd = np.zeros(nv)
        # N0 dynamically consistent
        if Jc is not None and Jc.shape[0] > 0:
            Lambda_c = Jc @ A_inv @ Jc.T
            Jc_dyn = A_inv @ Jc.T @ np.linalg.inv(Lambda_c + 1e-10 * np.eye(Jc.shape[0]))
            N = np.eye(nv) - Jc_dyn @ Jc
        else:
            N = np.eye(nv)
        for J, Jdv, xdd_cmd in tasks:
            JN = J @ N
            Lambda = JN @ A_inv @ JN.T
            JN_dyn = A_inv @ JN.T @ np.linalg.inv(Lambda + 1e-10 * np.eye(JN.shape[0]))
            qdd = qdd + JN_dyn @ (xdd_cmd - Jdv - J @ qdd)
            N = N - JN_dyn @ JN
        return qdd

    def CalcTorque(self, context, output):
        q, v = self._update_plant_state(context)

        f_mpc = self.get_input_port(1).Eval(context)
        s_phi = self.get_input_port(2).Eval(context)
        body_pose_des = self.get_input_port(5).Eval(context)

        # 1. Dynamics terms: tau = A qdd + h - Jc^T f,
        #    h = C(q,v) v - tau_g(q)  (bias + gravity, paper's "b + g").
        A = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
        h = self.plant.CalcBiasTerm(
            self.plant_context
        ) - self.plant.CalcGravityGeneralizedForces(self.plant_context)

        # 3. Contact Jacobian of the stance feet (toe frames, world frame) - needed for N0 before qdd
        Jc_list = []
        for i in range(4):
            if s_phi[i] > 0.5:
                J_foot = self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context,
                    JacobianWrtVariable.kV,
                    self.foot_frames[i],
                    np.zeros(3),
                    self.plant.world_frame(),
                    self.plant.world_frame(),
                )
                Jc_list.append(J_foot)

        num_active_feet = len(Jc_list)
        if num_active_feet > 0:
            Jc = np.vstack(Jc_list)
        else:
            Jc = np.zeros((0, self.num_velocities))

        # 2. Desired generalized accelerations from the task hierarchy (with N0)
        cmd_vel = self.get_input_port(6).Eval(context) if self.get_input_port(6).HasValue(context) else None
        tasks = self._body_task_data(body_pose_des, cmd_vel=cmd_vel)
        foot_pos_des = self.get_input_port(3).Eval(context)
        foot_vel_des = self.get_input_port(4).Eval(context)
        foot_task = self._foot_task_data(s_phi, foot_pos_des, foot_vel_des, v)
        if foot_task is not None:
            tasks.append(foot_task)
        qdd_cmd = self._prioritized_qdd(tasks, A, Jc=Jc)

        # Reaction forces of the stance feet, in the same leg order as Jc
        f_mpc_active = []
        for i in range(4):
            if s_phi[i] > 0.5:
                f_mpc_active.extend(f_mpc[3 * i : 3 * i + 3])
        f_mpc_active = np.array(f_mpc_active)

        # 4. WBIC QP (paper, part 4):
        #    min  ||delta_fr||^2_Q1 + ||delta_f||^2_Q2
        #    s.t. S_f (A qdd + h) = S_f Jc^T f_r   (floating-base dynamics)
        #         qdd = qdd_cmd + [delta_f; 0],  f_r = f_mpc + delta_fr
        #         friction cone and normal-force limits on f_r
        prog = MathematicalProgram()

        delta_f = prog.NewContinuousVariables(6, "delta_f")
        prog.AddQuadraticErrorCost(self.q2_weight * np.eye(6), np.zeros(6), delta_f)

        if num_active_feet > 0:
            delta_fr = prog.NewContinuousVariables(num_active_feet * 3, "delta_fr")
            prog.AddQuadraticErrorCost(
                self.q1_weight * np.eye(num_active_feet * 3),
                np.zeros(num_active_feet * 3),
                delta_fr,
            )

            # Floating-base dynamics (linear equality in [delta_f; delta_fr]):
            # A[:6,:6] delta_f - Jc[:,:6]^T delta_fr
            #     = -A[:6,:] qdd_cmd - h[:6] + Jc[:,:6]^T f_mpc
            A_eq = np.hstack([A[:6, :6], -Jc[:, :6].T])
            b_eq = -A[:6, :] @ qdd_cmd - h[:6] + Jc[:, :6].T @ f_mpc_active
            prog.AddLinearEqualityConstraint(
                A_eq, b_eq, np.concatenate([delta_f, delta_fr])
            )

            # Friction pyramid and normal-force limits, per active foot
            A_friction = np.array(
                [
                    [1.0, 0.0, -self.mu],
                    [-1.0, 0.0, -self.mu],
                    [0.0, 1.0, -self.mu],
                    [0.0, -1.0, -self.mu],
                ]
            )
            delta_fr_lb = np.full(num_active_feet * 3, -self.max_delta_fr)
            delta_fr_ub = np.full(num_active_feet * 3, self.max_delta_fr)
            for i in range(num_active_feet):
                f0 = f_mpc_active[3 * i : 3 * i + 3]
                # |fx| <= mu fz, |fy| <= mu fz  (linear in delta_fr)
                prog.AddLinearConstraint(
                    A_friction,
                    np.full(4, -np.inf),
                    -A_friction @ f0,
                    delta_fr[3 * i : 3 * i + 3],
                )
                # fz_min <= fz <= fz_max folded into the delta box bounds
                delta_fr_lb[3 * i + 2] = max(-self.max_delta_fr, self.fz_min - f0[2])
                delta_fr_ub[3 * i + 2] = min(self.max_delta_fr, self.fz_max - f0[2])
            prog.AddBoundingBoxConstraint(delta_fr_lb, delta_fr_ub, delta_fr)
        else:
            # Flight phase: no contacts, relax only the base acceleration
            b_eq = -A[:6, :] @ qdd_cmd - h[:6]
            prog.AddLinearEqualityConstraint(A[:6, :6], b_eq, delta_f)

        result = Solve(prog)

        if not result.is_success():
            # Fallback (safety): no torque rather than a wrong torque
            output.SetFromVector(np.zeros(self.num_joints))
            return

        # 5. Final joint torque (paper, part 5)
        qdd = qdd_cmd.copy()
        qdd[:6] += result.GetSolution(delta_f)

        if num_active_feet > 0:
            fr_opt = f_mpc_active + result.GetSolution(delta_fr)
            tau_full = A @ qdd + h - Jc.T @ fr_opt
        else:
            tau_full = A @ qdd + h

        # Joint torques = rows 6: of the generalized force vector. For this
        # URDF the generalized-velocity joint order (FR, FL, HR, HL) matches
        # the actuator declaration order in simulation.py.
        tau_joints = tau_full[6:]

        tau_clipped = np.clip(tau_joints, -self.tau_max, self.tau_max)
        output.SetFromVector(tau_clipped)

    def CalcQCmd(self, context, output):
        # Prioritized position IK (paper, WBIC part 1)
        # q_cmd = q_joints + delta_q_joints  where delta_q solves e_i = J_i delta_q
        q, v = self._update_plant_state(context)
        s_phi = self.get_input_port(2).Eval(context)
        foot_pos_des = self.get_input_port(3).Eval(context)
        body_pose_des = self.get_input_port(5).Eval(context)
        cmd_vel = self.get_input_port(6).Eval(context) if self.get_input_port(6).HasValue(context) else None

        X_WB = self.plant.EvalBodyPoseInWorld(self.plant_context, self.body)
        p_b = X_WB.translation()
        R_b = X_WB.rotation()
        R_des = RotationMatrix(RollPitchYaw(body_pose_des[3:]))
        e_rot = RollPitchYaw(R_des.multiply(R_b.transpose())).vector()

        moving = (np.hypot(cmd_vel[0], cmd_vel[1]) > 1e-3) if cmd_vel is not None else False
        if moving:
            # Do not accumulate position error in x, y when walking
            e_pos = np.array([0.0, 0.0, body_pose_des[2] - p_b[2]])
        else:
            e_pos = body_pose_des[:3] - p_b

        J_body = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.body_frame,
            np.zeros(3),
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

        tasks = [
            (J_body[:3], e_rot),
            (J_body[3:], e_pos),
        ]

        swing_idx = [i for i in range(4) if s_phi[i] < 0.5]
        if swing_idx:
            J_list = []
            e_list = []
            for i in swing_idx:
                J_f = self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context,
                    JacobianWrtVariable.kV,
                    self.foot_frames[i],
                    np.zeros(3),
                    self.plant.world_frame(),
                    self.plant.world_frame(),
                )
                p_cur = self.plant.CalcPointsPositions(
                    self.plant_context,
                    self.foot_frames[i],
                    np.zeros((3, 1)),
                    self.plant.world_frame(),
                ).ravel()
                p_des = foot_pos_des[3 * i : 3 * i + 3]
                J_list.append(J_f)
                e_list.append(p_des - p_cur)
            tasks.append((np.vstack(J_list), np.concatenate(e_list)))

        # N0 from contacts for IK (Kim IV.A (20))
        Jc_list = []
        for i in range(4):
            if s_phi[i] > 0.5:
                Jc_list.append(
                    self.plant.CalcJacobianTranslationalVelocity(
                        self.plant_context,
                        JacobianWrtVariable.kV,
                        self.foot_frames[i],
                        np.zeros(3),
                        self.plant.world_frame(),
                        self.plant.world_frame(),
                    )
                )
        Jc = np.vstack(Jc_list) if Jc_list else np.zeros((0, self.num_velocities))
        delta_q = self._prioritized_ik(tasks, Jc=Jc)
        q_joints = q[7 : 7 + self.num_joints]
        delta_joints = delta_q[6 : 6 + self.num_joints]

        q_cmd = q_joints + delta_joints
        # Regularize abduction joints towards 0.0 nominal so they do not drift or splay/cross
        # << nominal (all knees back) - abduction 0, hip -0.804, knee 1.607
        q_nom = np.array([
            0.0, -0.804,  1.607,   # FR
            0.0, -0.804,  1.607,   # FL
            0.0, -0.804,  1.607,   # HR
            0.0, -0.804,  1.607,   # HL
        ])
        for idx in [0, 3, 6, 9]:
            q_cmd[idx] = q_nom[idx] + delta_joints[idx]

        output.SetFromVector(q_cmd)

    def CalcVCmd(self, context, output):
        # Prioritized velocity IK (paper, WBIC part 2)
        # qdot_i = qdot_{i-1} + J_{i|pre}^\dagger (xdot_i_des - J_i qdot_{i-1})
        _, v = self._update_plant_state(context)
        s_phi = self.get_input_port(2).Eval(context)
        foot_vel_des = self.get_input_port(4).Eval(context)
        body_pose_des = self.get_input_port(5).Eval(context)
        cmd_vel = self.get_input_port(6).Eval(context) if self.get_input_port(6).HasValue(context) else None

        if cmd_vel is not None:
            vx_cmd, vy_cmd, wz_cmd = float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2])
        else:
            vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0

        yaw = body_pose_des[5]
        c, s = np.cos(yaw), np.sin(yaw)
        v_world_des = np.array([c * vx_cmd - s * vy_cmd, s * vx_cmd + c * vy_cmd, 0.0])
        w_world_des = np.array([0.0, 0.0, wz_cmd])

        J_body = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.body_frame,
            np.zeros(3),
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

        # Body tasks track commanded velocity, foot task tracks foot_vel_des
        tasks = [
            (J_body[:3], w_world_des),
            (J_body[3:], v_world_des),
        ]

        swing_idx = [i for i in range(4) if s_phi[i] < 0.5]
        if swing_idx:
            J_list = []
            xd_list = []
            for i in swing_idx:
                J_f = self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context,
                    JacobianWrtVariable.kV,
                    self.foot_frames[i],
                    np.zeros(3),
                    self.plant.world_frame(),
                    self.plant.world_frame(),
                )
                xd = foot_vel_des[3 * i : 3 * i + 3]
                J_list.append(J_f)
                xd_list.append(xd)
            tasks.append((np.vstack(J_list), np.concatenate(xd_list)))

        Jc_list = []
        for i in range(4):
            if s_phi[i] > 0.5:
                Jc_list.append(
                    self.plant.CalcJacobianTranslationalVelocity(
                        self.plant_context,
                        JacobianWrtVariable.kV,
                        self.foot_frames[i],
                        np.zeros(3),
                        self.plant.world_frame(),
                        self.plant.world_frame(),
                    )
                )
        Jc = np.vstack(Jc_list) if Jc_list else np.zeros((0, self.num_velocities))
        qdot = self._prioritized_ik(tasks, Jc=Jc)
        output.SetFromVector(qdot[6 : 6 + self.num_joints])
