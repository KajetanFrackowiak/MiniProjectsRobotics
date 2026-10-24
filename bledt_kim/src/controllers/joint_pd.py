import numpy as np
from pydrake.systems.framework import LeafSystem


class JointPDController(LeafSystem):

    def __init__(self, num_joints, kp, kd, max_torque=None, use_v_des=True):
        super().__init__()

        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.max_torque = max_torque
        self.n = num_joints
        self.use_v_des = use_v_des

        self.q_desired_input = self.DeclareVectorInputPort("q_desired", self.n)
        self.q_actual_input = self.DeclareVectorInputPort("q_actual", self.n)
        self.v_actual_input = self.DeclareVectorInputPort("v_actual", self.n)
        if self.use_v_des:
            self.v_desired_input = self.DeclareVectorInputPort("v_desired", self.n)
        else:
            self.v_desired_input = None

        self.DeclareVectorOutputPort("torque", self.n, self.CalcTorque)

    def CalcTorque(self, context, output):
        q_des = self.q_desired_input.Eval(context)
        q_act = self.q_actual_input.Eval(context)
        v_act = self.v_actual_input.Eval(context)
        if self.use_v_des and self.v_desired_input.HasValue(context):
            v_des = self.v_desired_input.Eval(context)
            torque = self.kp * (q_des - q_act) + self.kd * (v_des - v_act)
        else:
            # fallback: damping only (paper low-level uses -kd*v)
            torque = self.kp * (q_des - q_act) - self.kd * v_act
        if self.max_torque is not None:
            torque = np.clip(torque, -self.max_torque, self.max_torque)

        output.SetFromVector(torque)
