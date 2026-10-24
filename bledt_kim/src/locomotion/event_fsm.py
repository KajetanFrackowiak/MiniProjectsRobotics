import numpy as np
from pydrake.systems.framework import LeafSystem


class EventFSM(LeafSystem):
    """Event-Based FSM (Bledt 2018 sec 5) Contact_fustion.md:83.

    Inputs:
      s_phi (4): scheduled
      s_hat (4): estimated binary
      foot_pos_des (12): nominal from trajectory generator
      foot_vel_des (12): nominal vel
      foot_p_act (12): actual foot positions world (optional, not used)

    Outputs:
      s_eff (4): effective contact for WBIC/MPC (after FSM)
      foot_pos_mod (12): modified foot position for WBIC swing task
      foot_vel_mod (12): modified vel (0 when holding)

    States:
      hold_pos (12): latched position for Early
      early_timer (4): time since early detected
      late_timer (4): time since late detected
    Modes per leg:
      0 Swing (0,0)
      1 Contact (1,1)
      2 Early (0,1) -> hold, after delay -> Contact
      3 Late (1,0) -> extend down
    """

    def __init__(self, dt=0.001, t_delay=0.02, t_late_grace=0.04, v_extend=0.05):
        super().__init__()
        self.dt = dt
        self.t_delay = t_delay
        self.t_late_grace = t_late_grace
        self.v_extend = v_extend

        self.DeclareVectorInputPort("s_phi", 4)
        self.DeclareVectorInputPort("s_hat", 4)
        self.DeclareVectorInputPort("foot_pos_des", 12)
        self.DeclareVectorInputPort("foot_vel_des", 12)
        # optional actual foot not needed

        # discrete state: hold_pos 12 + early_t 4 + late_t 4 = 20
        self.DeclareDiscreteState(20)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.dt, offset_sec=0.0, update=self.DiscreteUpdate
        )

        self.DeclareVectorOutputPort(
            "s_eff", 4, self.CalcSEff, prerequisites_of_calc={self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "foot_pos_mod", 12, self.CalcPos, prerequisites_of_calc={self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "foot_vel_mod", 12, self.CalcVel, prerequisites_of_calc={self.xd_ticket()}
        )

        self._last_s_eff = np.zeros(4)
        self._last_pos = np.zeros(12)
        self._last_vel = np.zeros(12)

    def DiscreteUpdate(self, context, discrete_state):
        try:
            s_phi = self.get_input_port(0).Eval(context)
            s_hat = self.get_input_port(1).Eval(context)
            pos_des = self.get_input_port(2).Eval(context)
            vel_des = self.get_input_port(3).Eval(context)
        except Exception:
            return

        state = discrete_state.get_vector(0).CopyToVector()
        hold = state[0:12]
        early_t = state[12:16]
        late_t = state[16:20]

        # init hold to current pos if zero
        if np.allclose(hold, 0):
            hold = pos_des.copy()

        s_eff = np.zeros(4)
        pos_mod = pos_des.copy()
        vel_mod = vel_des.copy()

        for i in range(4):
            s = float(s_phi[i] > 0.5)
            h = float(s_hat[i] > 0.5)
            idx = slice(3 * i, 3 * i + 3)

            if s == 0 and h == 0:
                # Swing
                s_eff[i] = 0
                early_t[i] = 0
                late_t[i] = 0
                hold[idx] = pos_des[idx]
                pos_mod[idx] = pos_des[idx]
                vel_mod[idx] = vel_des[idx]

            elif s == 1 and h == 1:
                # Contact
                s_eff[i] = 1
                early_t[i] = 0
                late_t[i] = 0
                hold[idx] = pos_des[idx]
                pos_mod[idx] = pos_des[idx]
                vel_mod[idx] = np.zeros(3)  # stance vel 0

            elif s == 0 and h == 1:
                # Early
                early_t[i] += self.dt
                late_t[i] = 0
                # latch hold on first detection
                if early_t[i] <= self.dt + 1e-9:
                    hold[idx] = pos_des[idx]  # could be current actual pos but use des
                # hold position
                pos_mod[idx] = hold[idx]
                vel_mod[idx] = np.zeros(3)
                # after delay, switch to contact
                if early_t[i] > self.t_delay:
                    s_eff[i] = 1
                else:
                    s_eff[i] = 0  # still not trusted for force
                # keep hold
            else:  # s==1 and h==0  Late / landing transition
                late_t[i] += self.dt
                early_t[i] = 0
                if late_t[i] <= self.t_late_grace:
                    # Grace window at scheduled touchdown: maintain stance support
                    s_eff[i] = 1
                    hold[idx] = pos_des[idx]
                    pos_mod[idx] = pos_des[idx]
                    vel_mod[idx] = np.zeros(3)
                else:
                    # Persistent loss of contact in stance: genuine depression
                    s_eff[i] = 0  # inhibit force
                    pos_mod[idx] = hold[idx].copy() if np.any(hold[idx]) else pos_des[idx].copy()
                    # Bounded search extension downwards (max 3 cm below nominal)
                    z_target = pos_mod[3 * i + 2] - self.v_extend * self.dt
                    z_floor = pos_des[3 * i + 2] - 0.03
                    pos_mod[3 * i + 2] = max(z_target, z_floor)
                    pos_mod[3 * i + 0] = pos_des[3 * i + 0]
                    pos_mod[3 * i + 1] = pos_des[3 * i + 1]
                    vel_mod[idx] = np.array([0, 0, -self.v_extend])
                    hold[idx] = pos_mod[idx]

        # Safety: trot must have at least 2 supporting feet to avoid free fall
        if np.sum(s_eff) < 2:
            s_eff = s_phi.copy()

        new_state = np.concatenate([hold, early_t, late_t])
        discrete_state.get_mutable_vector(0).SetFromVector(new_state)
        self._last_s_eff = s_eff
        self._last_pos = pos_mod
        self._last_vel = vel_mod

    def _get_state(self, context):
        try:
            vec = context.get_discrete_state_vector().CopyToVector()
            hold = vec[0:12]
            s_eff = self._last_s_eff
            # need to recompute s_eff from _last? store
            # Instead compute from discrete + inputs? Use cached
            return s_eff
        except Exception:
            return self._last_s_eff

    def CalcSEff(self, context, output):
        # try discrete
        try:
            vec = context.get_discrete_state_vector().CopyToVector()
            # need inputs to compute? Use cached _last_s_eff updated in DiscreteUpdate
            # If context time just after update, cached is valid
            output.SetFromVector(self._last_s_eff)
        except Exception:
            output.SetFromVector(self._last_s_eff)

    def CalcPos(self, context, output):
        output.SetFromVector(self._last_pos)

    def CalcVel(self, context, output):
        output.SetFromVector(self._last_vel)
