import numpy as np
from pydrake.systems.framework import LeafSystem


class StateEstimator(LeafSystem):
    """Simple low-pass state estimator (WBC_MPC.md:21).

    Filters base_twist and pose so MPC / Raibert inputs stay free of noise.
    A single 15 Hz IIR pole is enough in simulation.

    Inputs:
      body_pose_rpy (6): [x, y, z, roll, pitch, yaw]
      body_twist (6): [wx, wy, wz, vx, vy, vz]

    Outputs:
      pose_filt (6)
      twist_filt (6)

    Discrete state: 6 pose + 6 twist
    """

    def __init__(self, dt=0.001, cutoff_hz=15.0):
        super().__init__()
        self.dt = dt
        self.cutoff = cutoff_hz
        # alpha = exp(-2*pi*cutoff*dt)
        import math
        self.alpha_twist = math.exp(-2 * math.pi * cutoff_hz * dt)
        # pose filter
        self.alpha_pose = math.exp(-2 * math.pi * 30.0 * dt)

        self.DeclareVectorInputPort("body_pose_rpy", 6)
        self.DeclareVectorInputPort("body_twist", 6)

        self.DeclareDiscreteState(12)  # 6 pose + 6 twist
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=dt, offset_sec=0.0, update=self.DiscreteUpdate
        )

        self.DeclareVectorOutputPort(
            "pose_filt", 6, self.CalcPose, prerequisites_of_calc={self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "twist_filt", 6, self.CalcTwist, prerequisites_of_calc={self.xd_ticket()}
        )

        self._last_pose = np.zeros(6)
        self._last_twist = np.zeros(6)

    def DiscreteUpdate(self, context, discrete_state):
        try:
            pose = self.get_input_port(0).Eval(context)
        except Exception:
            pose = self._last_pose
        try:
            twist = self.get_input_port(1).Eval(context)
        except Exception:
            twist = self._last_twist

        state = discrete_state.get_vector(0).CopyToVector()
        pose_f = state[0:6]
        twist_f = state[6:12]

        # init
        if np.allclose(pose_f, 0) and np.allclose(twist_f, 0):
            pose_f = pose.copy()
            twist_f = twist.copy()
        else:
            # pose: low-pass
            pose_f = self.alpha_pose * pose_f + (1 - self.alpha_pose) * pose
            # twist: low-pass
            twist_f = self.alpha_twist * twist_f + (1 - self.alpha_twist) * twist

        discrete_state.get_mutable_vector(0).SetFromVector(np.concatenate([pose_f, twist_f]))
        self._last_pose = pose_f
        self._last_twist = twist_f

    def CalcPose(self, context, output):
        try:
            vec = context.get_discrete_state_vector().CopyToVector()
            output.SetFromVector(vec[0:6])
        except Exception:
            output.SetFromVector(self._last_pose)

    def CalcTwist(self, context, output):
        try:
            vec = context.get_discrete_state_vector().CopyToVector()
            output.SetFromVector(vec[6:12])
        except Exception:
            output.SetFromVector(self._last_twist)
