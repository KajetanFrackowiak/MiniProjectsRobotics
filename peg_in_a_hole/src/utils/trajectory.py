
import numpy as np
from typing import Literal
from pydrake.trajectories import PiecewisePolynomial


PEG_LENGTH = 0.2  # meters

def get_peg_insertion_trajectory(
    nominal_pos_xz: np.ndarray,
    peg_frame_offset: float = 0.0,
    trans_pert: float = 0.0,
    rot_pert: float = 0.00,
) -> tuple[np.ndarray, PiecewisePolynomial]:

    x_nom, z_nom = nominal_pos_xz
    pos_initial = np.array(
        [x_nom, z_nom + peg_frame_offset, 0.0]
    )  # (pos_x, pos_z, pitch)
    pert = np.random.uniform(
        [-trans_pert, -trans_pert, -rot_pert], [trans_pert, trans_pert, rot_pert]
    )
    start_pos = pos_initial + pert

    z_pos_end = -PEG_LENGTH / 2 + peg_frame_offset

    traj = PiecewisePolynomial.FirstOrderHold(
        [0, 5.0],
        np.array([start_pos, [x_nom, z_pos_end, 0]]).T,
    )
    return start_pos, traj


def get_peg_frame_offset(placement: Literal["back", "center", "tip"]) -> float:
    if placement == "center":
        return 0.0
    elif placement == "back":
        return PEG_LENGTH / 2
    else:  # tip
        OFFSET = PEG_LENGTH / 4  # Place the frame a bit ahead of the peg tip
        return -PEG_LENGTH / 2 - OFFSET


        