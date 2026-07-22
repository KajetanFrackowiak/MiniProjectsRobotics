import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.solvers import Solve

def solve_ik(plant: MultibodyPlant, pose_des: np.ndarray, trans_tol: float = 0.0, ang_tol: float = 0.0) -> np.ndarray:
    context = plant.CreateDefaultContext()
    """
        The solve_ik will:

        1. The optimizer starts with an initial guess:
        q^0 = q_seed

        2. For a candidate q, Drake evaluates forward kinematics internally:
        q^i -> FK -> p_peg(q^i), R_peg(q^i)

        3. Drake evaluates the constraints:
        position error:
            p_peg(q^i) - p_des

        orientation error:
            R_des^T R_peg(q^i)

        4. The optimizer updates q to reduce the constraint violation
        and minimize the cost:
            (q - q_seed)^T Q (q - q_seed)

        5. Steps 2-4 are repeated until the constraints are satisfied.

        6. Return:
            q_solution = [q1, q2, ..., q7]
    """
    # Creates IK optimization problem, Find q such that FK(q) = desired pose
    ik = InverseKinematics(plant, context)
    # decision variables: [q1,q2,...,q7] (our unknwonw that we want to find)
    q = ik.q() 
    prog = ik.prog()

    P = plant.GetFrameByName("peg_tip_frame") # End-effector frame
    W = plant.world_frame()                   # World frame

    x_des, z_des, pitch_des = pose_des
    peg_pos_3d = np.array([x_des, 0, z_des])            # 2D -> 3D
    pos_bound_3d = np.array([trans_tol, 10, trans_tol]) # position tolerance [dx,dy,dz]

    # p_lower <= p_W_Q(q) <= p_upper
    ik.AddPositionConstraint(
        frameB=P,
        p_BQ=np.zeros(3),
        frameA=W,
        p_AQ_lower=peg_pos_3d - pos_bound_3d,
        p_AQ_upper=peg_pos_3d + pos_bound_3d,
    )

    R_WP_des = RollPitchYaw([0, pitch_des, 0]).ToRotationMatrix()
    # theta(R^T_{des}, R(q)) <= theta_{tol}
    # Since we only care about theta_y
    # We find such R_des that is most similar to R_y(theta_des)

    ik.AddOrientationConstraint(
        frameAbar=W,
        R_AbarA=R_WP_des,
        frameBbar=P,
        R_BbarB=RotationMatrix(),
        theta_bound=ang_tol,
    )
    # Initial configuration
    q_seed = np.array([0.1, -1.2, 1.6])
    # Weight matrix [7x7] (values)
    Q = np.eye(plant.num_positions())
    # (q - q_seed)^T Q (q - q_seed)
    prog.AddQuadraticErrorCost(Q, q_seed, q)
    # Initial solver guess: q = q_seed
    prog.SetInitialGuess(q, q_seed)

    # min_q || q - q_seed||^2 Subject to:
    # position(q) = position_des
    # rientation(q) = orientation_des
    result = Solve(prog) # [q1,q2,...,q7]
    assert result.is_success()

    return result.GetSolution(q)
