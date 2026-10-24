import math
import numpy as np

from pydrake.systems.framework import LeafSystem
from pydrake.multibody.tree import JacobianWrtVariable


class ContactEstimator(LeafSystem):
    """Contact Model Fusion (Bledt 2018) - Contact_fustion.md sec 1-4.

    Inputs:
      x_full (19+18): [q(7+12), v(6+12)] from plant
      s_phi (4): scheduled contact 0/1
      phi (4): subphase [0,1)
      tau (12): joint torques applied (from TorqueLimiter)

    Outputs:
      P_hat (4): fused contact probability [0,1]
      s_hat (4): binary 0/1 (threshold)
      f_hat (12): estimated foot forces (only fz used for fusion, full 3 per leg for logging)
      P_pred/Pz/Pf debug in discrete state (optional)

    Discrete state:
      filt (18): low-pass filtered term for momentum observer
      P_hat_prev (4): previous fused probability (for logging)

    Observer params:
      lambda_cut ~ 10-25 Hz, dt = period (e.g. 0.002)
      gamma = exp(-lambda*dt), beta = (1-gamma)*gamma^{-1}/dt
    """

    def __init__(
        self,
        plant,
        robot_model,
        dt=0.002,
        lambda_cut=15.0,
        mu_zg=0.0,
        sigma_zg=0.10,
        mu_fc=30.0,
        sigma_fc=7.0,  # was 40/5, lowered to capture 20-30N stance on 2 legs
        sigma_phi=0.05,
        sigma_w=0.02,
        sigma_v1=0.10,  # for height prior
        sigma_v2=0.20,  # for force prior
        threshold=0.5,
    ):
        super().__init__()
        self.plant = plant
        self.model = robot_model
        self.dt = dt
        self.lambda_cut = lambda_cut
        self.gamma = math.exp(-lambda_cut * dt)
        self.beta = (1 - self.gamma) * (1.0 / self.gamma) / dt if dt > 1e-9 else 0.0

        self.mu_zg = mu_zg
        self.sigma_zg = sigma_zg
        self.mu_fc = mu_fc
        self.sigma_fc = sigma_fc
        self.sigma_phi = sigma_phi
        self.sigma_w = sigma_w
        self.sigma_v1 = sigma_v1
        self.sigma_v2 = sigma_v2
        self.threshold = threshold

        self.num_positions = plant.num_positions()
        self.num_velocities = plant.num_velocities()
        self.num_joints = 12

        # Precompute Kalman gain per leg (scalar, H=[1;1])
        # Sigma_w is variance, Sigma_v = diag(v1,v2)
        # K = Sigma_w * H^T * (H Sigma_w H^T + Sigma_v)^-1
        # H Sigma_w H^T = [[w,w],[w,w]]
        # M = [[w+v1, w],[w, w+v2]]
        w = sigma_w
        v1 = sigma_v1
        v2 = sigma_v2
        M = np.array([[w + v1, w], [w, w + v2]], dtype=float)
        invM = np.linalg.inv(M)
        # Sigma_w * H^T = [w, w] (1x2)
        Ht = np.array([[1, 1]], dtype=float)  # actually H^T is 1x2? Let's do row
        # K = w * [1,1] * invM  => 1x2
        self.K = w * np.array([1, 1], dtype=float) @ invM  # shape (2,)
        # K1, K2 scalar per leg
        # self.K = [K1, K2]

        self.plant_context = plant.CreateDefaultContext()
        self.foot_frames = [
            plant.GetFrameByName("toe_fr"),
            plant.GetFrameByName("toe_fl"),
            plant.GetFrameByName("toe_hr"),
            plant.GetFrameByName("toe_hl"),
        ]
        # Body for pose
        try:
            self.body = plant.GetBodyByName("body")
        except Exception:
            self.body = None

        # Ports
        self.DeclareVectorInputPort("x_full", self.num_positions + self.num_velocities)
        self.DeclareVectorInputPort("s_phi", 4)
        self.DeclareVectorInputPort("phi", 4)
        self.DeclareVectorInputPort("tau", self.num_joints)

        # Discrete state: filt (18) + P_hat (4)
        self.DeclareDiscreteState(18 + 4)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.dt, offset_sec=0.0, update=self.DiscreteUpdate
        )

        # Outputs depend only on discrete state (break algebraic loop)
        self.DeclareVectorOutputPort(
            "P_hat", 4, self.CalcP, prerequisites_of_calc={self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "s_hat", 4, self.CalcS, prerequisites_of_calc={self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "f_hat", 12, self.CalcF, prerequisites_of_calc={self.xd_ticket()}
        )
        # Debug: also individual priors
        self.DeclareVectorOutputPort(
            "P_pred", 4, self.CalcPred, prerequisites_of_calc={self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "P_z", 4, self.CalcPz, prerequisites_of_calc={self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "P_f", 4, self.CalcPf, prerequisites_of_calc={self.xd_ticket()}
        )

        # Cache for last estimates
        self._last_f = np.zeros(12)
        self._last_P = np.full(4, 0.5)
        self._last_pred = np.zeros(4)
        self._last_Pz = np.zeros(4)
        self._last_Pf = np.zeros(4)

    # --- helpers ---------------------------------------------------------
    def _erf(self, x):
        return math.erf(x)

    def _P_pred_per_leg(self, s_phi, phi):
        # Contact_fustion.md eq prediction model
        # uses erf, mu 0/1, sigma 0.05
        s = float(s_phi)
        bar_s = 1.0 - s
        # avoid domain issues
        mu_c0 = 0.0
        mu_c1 = 1.0
        mu_bar0 = 0.0
        mu_bar1 = 1.0
        sig = self.sigma_phi
        # term for s=1
        term_c = self._erf((phi - mu_c0) / (sig * math.sqrt(2))) + self._erf(
            (mu_c1 - phi) / (sig * math.sqrt(2))
        )
        term_bar = 2 + self._erf((mu_bar0 - phi) / (sig * math.sqrt(2))) + self._erf(
            (phi - mu_bar1) / (sig * math.sqrt(2))
        )
        P = 0.5 * (s * term_c + bar_s * term_bar)
        # clamp
        if P < 0:
            P = 0.0
        if P > 1:
            P = 1.0
        return P

    def _P_z_per_leg(self, pz):
        # pz world height, ground 0
        arg = (self.mu_zg - pz) / (self.sigma_zg * math.sqrt(2))
        P = 0.5 * (1 + math.erf(arg))
        return float(np.clip(P, 0, 1))

    def _P_f_per_leg(self, fz):
        arg = (fz - self.mu_fc) / (self.sigma_fc * math.sqrt(2))
        P = 0.5 * (1 + math.erf(arg))
        return float(np.clip(P, 0, 1))

    def _update_plant(self, q, v):
        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)

    # --- discrete update -------------------------------------------------
    def DiscreteUpdate(self, context, discrete_state):
        # Read inputs (with fallback if not connected)
        try:
            x = self.get_input_port(0).Eval(context)
        except Exception:
            discrete_state.get_mutable_vector(0).SetFromVector(
                np.concatenate([discrete_state.get_vector(0).CopyToVector()[:18], self._last_P])
            )
            return
        try:
            s_phi = self.get_input_port(1).Eval(context)
        except Exception:
            s_phi = np.ones(4) * 0.5
        try:
            phi = self.get_input_port(2).Eval(context)
        except Exception:
            phi = np.zeros(4)
        try:
            tau_j = self.get_input_port(3).Eval(context)
        except Exception:
            tau_j = np.zeros(self.num_joints)

        q = x[: self.num_positions]
        v = x[self.num_positions :]

        self._update_plant(q, v)

        # Generalized momentum p = M * v
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
        p = M @ v

        # Bias term: C(q,v)*v - tau_g; use CalcBiasTerm and gravity terms.
        # Paper: M*ddq + C*qdot + g = S^T*tau + tau_d
        # Drake: M*dv + bias = tau; use C^T*qdot - g.
        try:
            bias = self.plant.CalcBiasTerm(self.plant_context)  # C*v
            g_vec = self.plant.CalcGravityGeneralizedForces(self.plant_context)  # -g
            Cqdot_minus_g = bias - g_vec  # C*v + g(q)
        except Exception:
            Cqdot_minus_g = np.zeros(self.num_velocities)

        # S^T tau (joint torques expanded to generalized)
        tau_gen = np.zeros(self.num_velocities)
        # first 6 floating base unactuated 0, next 12 joints
        tau_gen[6 : 6 + self.num_joints] = tau_j

        # Observer input u = beta*p + S^T*tau + C^T*qdot - g
        u = self.beta * p + tau_gen + Cqdot_minus_g

        # low-pass filtered term
        filt_old = discrete_state.get_vector(0).CopyToVector()[:18]
        # If filt uninitialized (0), start with u
        if np.allclose(filt_old, 0):
            filt_old = u
        filt_new = self.gamma * filt_old + (1 - self.gamma) * u

        # disturbance estimate
        hat_tau = self.beta * p - filt_new

        # foot force per leg
        f_hat = np.zeros(12)
        Pz = np.zeros(4)
        Pf = np.zeros(4)
        Pred = np.zeros(4)
        Ph = np.zeros(4)

        for i in range(4):
            # Jacobian for foot
            J = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kV,
                self.foot_frames[i],
                np.zeros(3),
                self.plant.world_frame(),
                self.plant.world_frame(),
            )  # 3x18
            # Extract leg joint columns 6+i*3
            col = 6 + i * 3
            J_leg = J[:, col : col + 3]  # 3x3
            # disturbance torque for leg
            tau_leg = hat_tau[col : col + 3]
            # solve J_leg^T f = tau_leg  => f = inv(J_leg^T) tau_leg
            # Use damped pseudo
            try:
                JT = J_leg.T
                # regularize
                JT_reg = JT + 1e-6 * np.eye(3)
                f = np.linalg.solve(JT_reg, tau_leg)
            except Exception:
                try:
                    f = np.linalg.lstsq(J_leg.T, tau_leg, rcond=None)[0]
                except Exception:
                    f = np.zeros(3)
            # Filter small values: negative fz unrealistic set 0
            # Keep fz positive only? but allow negative for flight
            f_hat[3 * i : 3 * i + 3] = f
            fz = float(f[2])

            # foot height
            p_foot = self.plant.CalcPointsPositions(
                self.plant_context,
                self.foot_frames[i],
                np.zeros((3, 1)),
                self.plant.world_frame(),
            ).ravel()
            pz = float(p_foot[2])  # world z, ground 0, toe radius offset? toe center at ~0.015 when in contact
            # Compensate toe radius? p_foot is toe frame center, radius 0.015 below. But height prior expects foot bottom.
            # Keep as is, threshold will adapt.

            # priors
            P_pred = self._P_pred_per_leg(s_phi[i], phi[i])
            P_z = self._P_z_per_leg(pz)
            P_f = self._P_f_per_leg(fz)

            Pred[i] = P_pred
            Pz[i] = P_z
            Pf[i] = P_f

            # Kalman fusion per leg
            # P_hat = P_pred + K1*(Pz - P_pred) + K2*(Pf - P_pred)
            z = np.array([P_z, P_f])
            P_hat = P_pred + self.K[0] * (P_z - P_pred) + self.K[1] * (P_f - P_pred)
            Ph[i] = float(np.clip(P_hat, 0, 1))

        # store
        # discrete_state vector 22: first 18 filt, next 4 P_hat
        new_state = np.concatenate([filt_new, Ph])
        discrete_state.get_mutable_vector(0).SetFromVector(new_state)
        self._last_f = f_hat
        self._last_P = Ph
        self._last_pred = Pred
        self._last_Pz = Pz
        self._last_Pf = Pf

    # --- outputs ---------------------------------------------------------
    def _get_discrete(self, context):
        try:
            vec = context.get_discrete_state_vector().CopyToVector()
            filt = vec[:18]
            Ph = vec[18:22]
            return filt, Ph
        except Exception:
            try:
                vec = context.get_discrete_state(0).CopyToVector()
                return vec[:18], vec[18:22]
            except Exception:
                return None, self._last_P

    def CalcP(self, context, output):
        _, Ph = self._get_discrete(context)
        if Ph is None or np.allclose(Ph, 0):
            Ph = self._last_P
        output.SetFromVector(Ph)

    def CalcS(self, context, output):
        _, Ph = self._get_discrete(context)
        if Ph is None:
            Ph = self._last_P
        s = (Ph >= self.threshold).astype(float)
        output.SetFromVector(s)

    def CalcF(self, context, output):
        # return last f_hat 12
        output.SetFromVector(self._last_f)

    def CalcPred(self, context, output):
        output.SetFromVector(self._last_pred)

    def CalcPz(self, context, output):
        output.SetFromVector(self._last_Pz)

    def CalcPf(self, context, output):
        output.SetFromVector(self._last_Pf)
