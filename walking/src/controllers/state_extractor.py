import numpy as np

from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.common.eigen_geometry import Quaternion
from pydrake.systems.framework import LeafSystem


class JointStateExtractor(LeafSystem):
    def __init__(self, num_positions, num_velocities, num_joints):
        super().__init__()

        self.num_positions = num_positions
        self.num_velocities = num_velocities
        self.num_joints = num_joints

        # Single input port for full continuous state vector (q + v)
        self.DeclareVectorInputPort("x_full", num_positions + num_velocities)

        # Joint-only states (for Low-level PD Controller)
        self.DeclareVectorOutputPort("q_joints", num_joints, self.CalcQJoints)
        self.DeclareVectorOutputPort("v_joints", num_joints, self.CalcVJoints)

        # Base states (for future MPC / WBC)
        self.DeclareVectorOutputPort("body_pose", 7, self.CalcBodyPose)
        self.DeclareVectorOutputPort("body_twist", 6, self.CalcBodyTwist)
        # [x, y, z, roll, pitch, yaw] (for trajectory generator / MPC)
        self.DeclareVectorOutputPort("body_pose_rpy", 6, self.CalcBodyPoseRpy)

    def CalcQJoints(self, context, output):
        x = self.get_input_port(0).Eval(context)
        # Joint positions start after 7 floating base coordinates (quaternion + xyz)
        output.SetFromVector(x[7 : 7 + self.num_joints])

    def CalcVJoints(self, context, output):
        x = self.get_input_port(0).Eval(context)
        # Joint velocities start after 6 floating base velocities (angular + linear)
        v_start = self.num_positions
        output.SetFromVector(x[v_start + 6 : v_start + 6 + self.num_joints])

    def CalcBodyPose(self, context, output):
        x = self.get_input_port(0).Eval(context)
        output.SetFromVector(x[0:7])

    def CalcBodyPoseRpy(self, context, output):
        x = self.get_input_port(0).Eval(context)
        # Drake floating-base quaternion ordering: [qw, qx, qy, qz, x, y, z]
        q = np.asarray(x[0:4], dtype=float)
        n = np.linalg.norm(q)
        if n < 1e-6 or np.isnan(n):
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q = q / n
        try:
            quat = Quaternion(q[0], q[1], q[2], q[3])
            quat.Normalize()
            rpy = RollPitchYaw(RotationMatrix(quat)).vector()
        except Exception:
            rpy = np.zeros(3)
        output.SetFromVector(np.concatenate([x[4:7], rpy]))

    def CalcBodyTwist(self, context, output):
        x = self.get_input_port(0).Eval(context)
        v_start = self.num_positions
        output.SetFromVector(x[v_start : v_start + 6])
