import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from controllers.iiwa_controller import CreateIiwaPlant


def create_q_knots(
    pose_lst: list[RigidTransform], t_lst: np.ndarray | None = None
) -> np.ndarray:
    q_knots = []
    plant, _ = CreateIiwaPlant()
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")
    forearm_frame = plant.GetBodyByName("iiwa_link_5").body_frame()
    q_nominal = np.array(
        [0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0, 0.0, 0.0]
    )  # nominal joint angles for joint-centering.

    pos_tol = 0.001
    ori_tol = 0.01 * np.pi

    W = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    for i in range(len(pose_lst)):
        ik = InverseKinematics(plant)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()  # Get MathematicalProgram

        X_WG = pose_lst[i]
        p_WG = X_WG.translation()
        R_WG = X_WG.rotation()

        if t_lst is not None and 21.0 <= t_lst[i] <= 27.0:
            # Keep the forearm above the open right-door panel during the
            # left-door entry (panel top edge is at y ~ -0.15).
            ik.AddPositionConstraint(
                frameB=forearm_frame,
                p_BQ=np.zeros(3),
                frameA=world_frame,
                p_AQ_lower=[-10.0, -0.10, -10.0],
                p_AQ_upper=[10.0, 10.0, 10.0],
            )

        # 1. Initial guess
        if i == 0:
            prog.SetInitialGuess(q_variables, q_nominal)
        else:
            prog.SetInitialGuess(q_variables, q_knots[-1])

        # 2. Position constraint: gripper origin in world within +- 1mm
        ik.AddPositionConstraint(
            frameB=gripper_frame,
            p_BQ=np.zeros(
                3
            ),  # Q is the origin of the gripper frame, p_BQ is a vector from the
            # gripper frame to the point we want to constrain (the origin in this case).
            frameA=world_frame,
            p_AQ_lower=p_WG - pos_tol,
            p_AQ_upper=p_WG + pos_tol,
        )

        # 3. Orientation constraint: gripper orientation in world within +- 0.01*pi rad.
        ik.AddOrientationConstraint(
            frameAbar=gripper_frame,
            R_AbarA=RotationMatrix.Identity(),  # A = gripper
            frameBbar=world_frame,
            R_BbarB=R_WG,  # desired orientation of gripper in world
            theta_bound=ori_tol,
        )

        # 4. Joint-centering cost to help guide the solver towards good solutions.
        prog.AddQuadraticErrorCost(W, q_nominal, q_variables)

        result = Solve(prog)

        if not result.is_success():
            # Retry with the nominal seed (the chain seed can trap the solver
            # in a local minimum even for feasible poses).
            prog.SetInitialGuess(q_variables, q_nominal)
            result = Solve(prog)

        assert result.is_success()

        q_knots.append(result.GetSolution(q_variables))

    return q_knots
