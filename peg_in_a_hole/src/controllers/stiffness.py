import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import BasicVector, Context, LeafSystem


class StiffnessController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, k_p: np.ndarray, k_d: np.ndarray) -> None:
        super().__init__()

        assert k_p.shape == (3,)
        assert k_d.shape == (3,)
        self._plant = plant
        self._K_p = np.diag(k_p)
        self._K_d = np.diag(k_d)

        self._plant_context = plant.CreateDefaultContext()

        self._q_des_port = self.DeclareVectorInputPort(
            "q_des", 3
        )  # [x_des, z_des, theta_des]
        self._q_dot_des_port = self.DeclareVectorInputPort(
            "q_dot_des", 3
        )  # [x_dot_des, z_dot_des, theta_dot_des]
        self._state_port = self.DeclareVectorInputPort(
            "state", 6
        )  # [x, z, theta, x_dot, z_dot, theta_dot]

        self.DeclareVectorOutputPort(
            "actuation", 3, self.OutputForces
        )  # [Fx, Fz, tau_theta]

    def OutputForces(self, context: Context, output: BasicVector) -> None:
        # We first take the state from the controller's context
        # (it will be also used to update the plant's context to get the
        # current gravity forces)
        x = self._state_port.Eval(context)
        q = x[:3]  # [x, z, theta]
        q_dot = x[3:]  # [x_dot, z_dot, theta_dot]

        q_des = self._q_des_port.Eval(context)  # [x_des, z_des, theta_des]
        q_dot_des = self._q_dot_des_port.Eval(
            context
        )  # [x_dot_des, z_dot_des, theta_dot_des]

        # Since MultibodyPlant does not read local variables and instead of it,
        # it always performs calculations using the state stored in its Context,
        # we first update our plant's Context with the state from the
        # controller's Context
        self._plant.SetPositionsAndVelocities(self._plant_context, x)
        tau_gravity = self._plant.CalcGravityGeneralizedForces(self._plant_context)

        u = self._K_p @ (q_des - q) + self._K_d @ (q_dot_des - q_dot) - tau_gravity

        output.SetFromVector(u)
