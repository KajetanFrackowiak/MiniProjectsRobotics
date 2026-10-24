import numpy as np
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import LeafSystem


class MPCController(LeafSystem):
    """Full horizon MPC (Kim 2019) - WBC_MPC.md sec MPC.

    Horizon QP:
      min sum ||x_{k+1}-x_ref||_Q^2 + ||f_k||_R^2
      s.t. x_{k+1}=A x_k + B_k f_k + g, friction, swing zero
    State x=[Theta(3), p(3), omega(3), v(3)] 12
    f=[f0..f3] 12 per step (3 per foot)
    Rate 40 Hz, N=10, dt=0.025
    """

    def __init__(
        self,
        plant,
        robot_model,
        period=0.5,
        duty_factor=0.65,
        mu=0.6,
        fz_min=5.0,
        fz_max=180.0,
        horizon=10,
        dt=0.025,
        hip_offsets=None,
    ):
        super().__init__()
        self.plant = plant
        self.model = robot_model
        self.mass = plant.CalcTotalMass(plant.CreateDefaultContext(), [robot_model])
        self.g = 9.81
        self.mu = mu
        self.fz_min = fz_min
        self.fz_max = fz_max
        self.period = period
        self.duty = duty_factor
        self.N = horizon
        self.dt = dt
        # Hip offsets in body frame (same as LegTrajectoryGenerator)
        if hip_offsets is None:
            self.hip_offsets = [
                np.array([0.19, -0.111]),
                np.array([0.19, 0.111]),
                np.array([-0.19, -0.111]),
                np.array([-0.19, 0.111]),
            ]
        else:
            self.hip_offsets = [np.array(o) for o in hip_offsets]

        # Inertia approx (body). Try to get from plant, fallback diag
        try:
            # Use default body inertia
            body = plant.GetBodyByName("body")
            # Try to compute spatial inertia at default context
            ctx0 = plant.CreateDefaultContext()
            # Drake's CalcSpatialInertia not directly, approximate
            # Use fixed value
            # Mini Cheetah full-body inertia about COM in nominal posture
            self.I_body = np.diag([0.14, 0.32, 0.33])
            self.I_inv = np.linalg.inv(self.I_body)
        except Exception:
            self.I_body = np.diag([0.14, 0.32, 0.33])
            self.I_inv = np.linalg.inv(self.I_body)

        # Weights - Kim et al. 2019 / MIT Cheetah standard tuning
        self.Q = np.diag(
            [100, 80, 20,  # Theta roll pitch yaw
             10, 15, 150,  # p x y z
             3, 2, 10,  # omega
             20, 30, 10]  # v
        )
        self.R = np.eye(12) * 1e-4

        self.phase_offset = np.array([0.0, 0.5, 0.5, 0.0])

        self.DeclareVectorInputPort("base_pose", 6)
        self.DeclareVectorInputPort("base_twist", 6)
        self.DeclareVectorInputPort("s_phi", 4)
        self.DeclareVectorInputPort("cmd_vel", 3)

        self.mpc_period = dt  # 0.025
        self.DeclareDiscreteState(12)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.mpc_period, offset_sec=0.0, update=self.DiscreteUpdate
        )
        self.DeclareVectorOutputPort("f_mpc", 12, self.CalcOutput)

        # For warm start
        self.last_f = np.array([0, 0, self.mass * 9.81 / 4] * 4, dtype=float)

    def _r_foot(self, base_pose, yaw, h):
        # world r vectors COM->foot (approx hip + down)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        rs = []
        for hip in self.hip_offsets:
            r_xy = R @ hip
            # foot approx below hip by h
            r = np.array([r_xy[0], r_xy[1], -h])
            rs.append(r)
        return rs

    def _build_AB(self, yaw, rs):
        dt = self.dt
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        A = np.eye(12)
        # Theta += Rz*omega*dt
        A[0:3, 6:9] = Rz * dt
        # p += v*dt
        A[3:6, 9:12] = np.eye(3) * dt
        B = np.zeros((12, 12))
        # GI = Rz * BI * Rz^T, GI_inv = Rz * BI_inv * Rz^T (Kim III.A (5))
        GI_inv = Rz @ self.I_inv @ Rz.T
        for i, r in enumerate(rs):
            # v += f/m * dt
            B[9:12, 3 * i: 3 * i + 3] = np.eye(3) * dt / self.mass
            # omega += GI_inv * (r x f) * dt
            skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            B[6:9, 3 * i: 3 * i + 3] = GI_inv @ skew * dt
        g_hat = np.zeros(12)
        g_hat[11] = -self.g * dt  # v_z gravity
        return A, B, g_hat

    def DiscreteUpdate(self, context, discrete_state):
        t = context.get_time()
        base_pose = self.get_input_port(0).Eval(context)
        base_twist = self.get_input_port(1).Eval(context)
        s_phi_now = self.get_input_port(2).Eval(context)
        cmd_vel = self.get_input_port(3).Eval(context)

        # Current state x0
        # base_pose [x,y,z,roll,pitch,yaw]
        x0 = np.zeros(12)
        # Theta
        x0[0:3] = base_pose[3:6]
        # p
        x0[3:6] = base_pose[0:3]
        # omega
        x0[6:9] = base_twist[0:3]
        # v
        x0[9:12] = base_twist[3:6]

        yaw = base_pose[5]
        c, s = np.cos(yaw), np.sin(yaw)
        h = base_pose[2]
        if h < 0.1:
            h = 0.285
        rs = self._r_foot(base_pose, yaw, h)
        A, B, g_hat = self._build_AB(yaw, rs)
        vx_cmd, vy_cmd, wz_cmd = float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2])
        # Commanded velocity in world frame
        vx_world = c * vx_cmd - s * vy_cmd
        vy_world = s * vx_cmd + c * vy_cmd

        # Use current p as start, propagate with cmd; lock roll/pitch/yaw to nominal
        x_ref = []
        for k in range(1, self.N + 1):
            xr = np.zeros(12)
            # Theta ref: roll/pitch zero, yaw tracks commanded rate towards 0 heading
            xr[0] = 0.0  # roll des 0
            xr[1] = 0.0  # pitch des 0
            xr[2] = wz_cmd * k * self.dt  # target yaw
            # p ref: x,y move with world velocity, z keep nominal height
            xr[3] = base_pose[0] + vx_world * k * self.dt
            xr[4] = base_pose[1] + vy_world * k * self.dt
            xr[5] = 0.285
            # omega ref
            xr[6:9] = [0.0, 0.0, wz_cmd]
            # v ref (in world frame)
            xr[9] = vx_world
            xr[10] = vy_world
            xr[11] = 0.0
            x_ref.append(xr)
        x_ref = np.array(x_ref)  # N x12

        # Predict s_phi horizon
        s_phi_horizon = []
        for k in range(self.N):
            tk = t + k * self.dt
            phi = np.mod(tk / self.period + self.phase_offset, 1.0)
            s_phi_k = (phi < self.duty).astype(float)
            s_phi_horizon.append(s_phi_k)

        # Build QP
        prog = MathematicalProgram()
        # f variables: N*12
        f_vars = prog.NewContinuousVariables(self.N * 12, "f")
        # x variables: N*12 (x1..xN)
        x_vars = prog.NewContinuousVariables(self.N * 12, "x")

        # Cost and constraints
        # For each horizon step k
        A_list = []
        for k in range(self.N):
            xk = x_vars[k * 12:(k + 1) * 12]
            fk = f_vars[k * 12:(k + 1) * 12]
            # Cost: ||x - x_ref||_Q
            e = xk - x_ref[k]
            prog.AddQuadraticCost(e.dot(self.Q.dot(e)))
            prog.AddQuadraticCost(fk.dot(self.R.dot(fk)))
            # Friction / swing
            s_phi_k = s_phi_horizon[k]
            for i in range(4):
                fi = fk[3 * i: 3 * i + 3]
                if s_phi_k[i] < 0.5:
                    # swing: f=0
                    prog.AddLinearEqualityConstraint(fi[0] == 0)
                    prog.AddLinearEqualityConstraint(fi[1] == 0)
                    prog.AddLinearEqualityConstraint(fi[2] == 0)
                else:
                    # stance: friction pyramid and fz limits
                    # |fx| <= mu fz  -> fx - mu fz <=0 and -fx -mu fz <=0 similarly fy
                    # Linear: [1 0 -mu] [fx fy fz]^T <=0
                    # Use AddLinearConstraint with bounds
                    # Use two constraints for x and y
                    prog.AddLinearConstraint(fi[0] - self.mu * fi[2] <= 0)
                    prog.AddLinearConstraint(-fi[0] - self.mu * fi[2] <= 0)
                    prog.AddLinearConstraint(fi[1] - self.mu * fi[2] <= 0)
                    prog.AddLinearConstraint(-fi[1] - self.mu * fi[2] <= 0)
                    prog.AddBoundingBoxConstraint(self.fz_min, self.fz_max, fi[2])

        # Dynamics constraints: x_{k+1}=A x_k + B f_k + g
        for k in range(self.N):
            xk1 = x_vars[k * 12:(k + 1) * 12]
            fk = f_vars[k * 12:(k + 1) * 12]
            if k == 0:
                x_prev = x0
                rhs = A.dot(x_prev) + g_hat
                for r in range(12):
                    expr = xk1[r] - B[r, :].dot(fk)
                    prog.AddLinearEqualityConstraint(expr == rhs[r])
            else:
                xk_prev = x_vars[(k - 1) * 12: k * 12]
                for r in range(12):
                    expr = xk1[r] - A[r, :].dot(xk_prev) - B[r, :].dot(fk)
                    prog.AddLinearEqualityConstraint(expr == g_hat[r])

        result = Solve(prog)
        if result.is_success():
            f_opt = result.GetSolution(f_vars)
            f0 = f_opt[0:12]
            # Clip just in case
            # Ensure swing legs zero (already)
            discrete_state.get_mutable_vector(0).SetFromVector(f0)
            self.last_f = f0
        else:
            # Fallback to last solution or static
            # Try to keep last_f if available
            if self.last_f is not None:
                discrete_state.get_mutable_vector(0).SetFromVector(self.last_f)
            else:
                fz = self.mass * self.g / max(1, int(np.sum(s_phi_now > 0.5)))
                f = np.zeros(12)
                for i in range(4):
                    if s_phi_now[i] > 0.5:
                        f[3 * i + 2] = fz
                discrete_state.get_mutable_vector(0).SetFromVector(f)

    def CalcOutput(self, context, output):
        try:
            f = context.get_discrete_state_vector().CopyToVector()
        except Exception:
            f = context.get_discrete_state(0).CopyToVector()
        if np.allclose(f, 0):
            fz = self.mass * self.g / 4.0
            f = np.array([0, 0, fz] * 4)
        output.SetFromVector(f)
