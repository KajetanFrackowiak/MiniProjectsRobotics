import numpy as np
from pydrake.systems.framework import LeafSystem, Context, BasicVector
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.math import RollPitchYaw

class OperationalStiffnessController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, k_p: np.ndarray, k_d: np.ndarray) -> None:
        super().__init__()

        self._plant = plant
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        assert k_p.shape == (3,)
        assert k_d.shape == (3,)

        self._K_p = np.diag(k_p)
        self._K_d = np.diag(k_d)

        self._torque_max = 8

        self._P = plant.GetFrameByName("peg_tip_frame") # End-effector frame
        self._W = plant.world_frame()                   # World frame

        
        self._free_joint_indices = [
            plant.GetJointByName(j).position_start()
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
        ] # [q2, q4, q6]

        self._plant_context = plant.CreateDefaultContext()

        self._p_des_port = self.DeclareVectorInputPort("p_des", 3)            # [x_des, z_des, theta_des]
        self._v_des_port = self.DeclareVectorInputPort("v_des", 3)            # [x_dot_dex, z_dot_des, theta_dot_des]

        self._iiwa_state_port = self.DeclareVectorInputPort("iiwa_state", 6)  # [q2, q4, q6, q2_dot, q4_dot, q6_dot]

        self.DeclareVectorOutputPort("iiwa_torques", 3, self.CalcTorqueOutput) # [tau_q2, tau_q4, tau_q6]


    def CalcTorqueOutput(self, context: Context, output: BasicVector) -> None:
        x_iiwa = self._iiwa_state_port.Eval(context)
        q = x_iiwa[:3]      # [q2, q4, q6]
        q_dot = x_iiwa[3:]  # [q2_dot, q4_dot, q6_dot]

        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        self._plant.SetVelocities(self._plant_context, self._iiwa, q_dot)

        p_des = self._p_des_port.Eval(context) # [x_des, z_des, theta_des]
        v_des = self._v_des_port.Eval(context) # [x_dot_des, z_dot_des, theta_dot_des]

        # Jv_WP: 6 x n 
        # rows [wx, wy, wz, vx, vy, vz]
        # cols [q1_dot, q2_dot, ..., q7_dot] 
        Jv_WP = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._P,                    # frame for which Jacobian is computed
            np.zeros(3),                # point offset [dx,dy,dz] = [0,0,0]
            self._W,                    # reference frame
            self._W                     # expressed-in frame
        )

        # For planar iiwa we select only vx, vz (velocities), wy (angular veolocity) 
        config_idxs = [3, 5, 1]
        # J: 3 x 3, Rows: [vx, vz, wy], Cols: [q2_dot, q4_dot, q6_dot]
        J = Jv_WP[np.ix_(config_idxs, self._free_joint_indices)]

        X_WP = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._P)
        p = np.zeros(3)
        p[0:2] = X_WP.translation()[[0, 2]]                 # [x, z]
        p[2] = RollPitchYaw(X_WP.rotation()).pitch_angle()  # [theta]

        v = J @ q_dot # [x_dot, z_dot, theta_dot]

        # [Fx, Fz, tau_theta] in task-space
        F_u = self._K_p @ (p_des - p) + self._K_d @ (v_des - v)
        # # [tau_2, tau_4, tau_6] in joint-space
        u = J.T @ F_u 

        u_clamped = np.clip(u, -self._torque_max, self._torque_max)

        output.SetFromVector(u_clamped)