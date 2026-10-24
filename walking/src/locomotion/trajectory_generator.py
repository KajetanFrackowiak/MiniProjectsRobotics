import numpy as np
from pydrake.systems.framework import LeafSystem


class LegTrajectoryGenerator(LeafSystem):
    """World-frame swing foot trajectory generator (Kim et al. 2019 / Bledt et al. 2018).

    At liftoff, latches actual foot position p0 in world frame.
    During swing, generates a smooth curve from p0 to the Raibert target pf:
      p_xy(s) = (1 - s_curve) * p0_xy + s_curve * pf_xy
      p_z(s)  = z_ground + h_step * sin(pi * s)
    where s in [0, 1] is the swing phase, and s_curve is a smooth cubic polynomial.
    """

    def __init__(
        self,
        step_height=0.045,
        duty_factor=0.5,
        period=0.5,
        ground_clearance=0.015,
    ):
        super().__init__()
        self.step_height = step_height
        self.duty_factor = duty_factor
        self.period = period
        self._period = period
        self.ground_clearance = ground_clearance

        self.hip_offsets = [
            np.array([0.19, -0.111]),  # FR
            np.array([0.19,  0.111]),  # FL
            np.array([-0.19, -0.111]), # HR
            np.array([-0.19,  0.111]), # HL
        ]

        self.DeclareVectorInputPort("s_phi", 4)
        self.DeclareVectorInputPort("phi", 4)
        self.DeclareVectorInputPort("foot_placement", 12)
        self.DeclareVectorInputPort("foot_act", 12)

        # Discrete state: latched liftoff positions in world frame (12 floats)
        self.DeclareDiscreteState(12)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.001, offset_sec=0.0, update=self.DiscreteUpdate
        )

        self.DeclareVectorOutputPort("foot_pos_des", 12, self.CalcFootPos)
        self.DeclareVectorOutputPort("foot_vel_des", 12, self.CalcFootVel)

    def DiscreteUpdate(self, context, discrete_state):
        s_phi = self.get_input_port(0).Eval(context)
        foot_act = self.get_input_port(3).Eval(context)
        liftoff = discrete_state.get_vector(0).CopyToVector()

        if np.allclose(liftoff, 0):
            liftoff = foot_act.copy()

        # For stance legs, keep updating liftoff position to current foot position
        for i in range(4):
            if s_phi[i] > 0.5:
                liftoff[3 * i : 3 * i + 3] = foot_act[3 * i : 3 * i + 3]

        discrete_state.get_mutable_vector(0).SetFromVector(liftoff)

    def CalcFootPos(self, context, output):
        s_phi = self.get_input_port(0).Eval(context)
        phi = self.get_input_port(1).Eval(context)
        placement = self.get_input_port(2).Eval(context)
        foot_act = self.get_input_port(3).Eval(context)
        liftoff = context.get_discrete_state(0).CopyToVector()
        if np.allclose(liftoff, 0):
            liftoff = foot_act.copy()

        D = self.duty_factor
        p_des = np.zeros(12)
        for i in range(4):
            if s_phi[i] > 0.5:
                # Stance foot: held at actual contact position on ground
                p_des[3 * i : 3 * i + 3] = foot_act[3 * i : 3 * i + 3]
                p_des[3 * i + 2] = self.ground_clearance
            else:
                # Swing foot: smooth interpolation from p_liftoff to p_target
                s = np.clip((phi[i] - D) / (1.0 - D), 0.0, 1.0)
                p0 = liftoff[3 * i : 3 * i + 3]
                pf = placement[3 * i : 3 * i + 3]
                s_curve = s * s * (3.0 - 2.0 * s)
                p_des[3 * i : 3 * i + 2] = (1.0 - s_curve) * p0[:2] + s_curve * pf[:2]
                p_des[3 * i + 2] = (
                    self.ground_clearance + self.step_height * np.sin(np.pi * s)
                )
        output.SetFromVector(p_des)

    def CalcFootVel(self, context, output):
        s_phi = self.get_input_port(0).Eval(context)
        phi = self.get_input_port(1).Eval(context)
        placement = self.get_input_port(2).Eval(context)
        foot_act = self.get_input_port(3).Eval(context)
        liftoff = context.get_discrete_state(0).CopyToVector()
        if np.allclose(liftoff, 0):
            liftoff = foot_act.copy()

        D = self.duty_factor
        period = getattr(self, "_period", 0.5)
        if hasattr(self, "period"):
            period = self.period
        T_swing = max(1e-4, (1.0 - D) * period)

        v_des = np.zeros(12)
        for i in range(4):
            if s_phi[i] > 0.5:
                v_des[3 * i : 3 * i + 3] = 0.0
            else:
                s = np.clip((phi[i] - D) / (1.0 - D), 0.0, 1.0)
                p0 = liftoff[3 * i : 3 * i + 3]
                pf = placement[3 * i : 3 * i + 3]
                ds_dt = 6.0 * s * (1.0 - s) / T_swing
                v_des[3 * i : 3 * i + 2] = (pf[:2] - p0[:2]) * ds_dt
                v_des[3 * i + 2] = (
                    self.step_height * np.pi * np.cos(np.pi * s) / T_swing
                )
        output.SetFromVector(v_des)

    def set_period(self, period: float):
        self.period = period
        self._period = period
