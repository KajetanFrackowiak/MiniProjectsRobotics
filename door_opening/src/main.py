import argparse

import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.trajectories import PiecewisePolynomial

from ik.ik import create_q_knots
from stations.station import teleop_inverse_kinematics
from trajectories.keyframes import add_poses
from trajectories.simulate_trajectory import build_and_simulate_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Simulate trajectory for manipulation station."
    )
    parser.add_argument(
        "--visualize-traj",
        action="store_true",
        help="Visualize the trajectory in Meshcat.",
    )
    parser.add_argument(
        "--run-teleop-ik",
        action="store_true",
        help="Run teleoperation inverse kinematics.",
    )
    parser.add_argument(
        "--run-ik",
        action="store_true",
        help="Run inverse kinematics to compute joint positions.",
    )
    args = parser.parse_args()

    meshcat = StartMeshcat()
    t_lst = np.linspace(0, 56, 50)

    if args.visualize_traj:
        _ = add_poses(meshcat, t_lst)
    elif args.run_teleop_ik:
        teleop_inverse_kinematics(meshcat)
    elif args.run_ik:
        pose_lst = add_poses(meshcat, t_lst, show_triads=False)
        q_knots = np.array(create_q_knots(pose_lst, t_lst))
        q_traj = PiecewisePolynomial.CubicShapePreserving(
            t_lst.tolist(), [row for row in q_knots[:, 0:7]]
        )

        gripper_t_lst = np.array(
            [
                0.0,
                5.0,
                6.0,
                10.0,
                11.0,
                26.0,
                27.0,
                32.0,
                33.0,
                44.0,
                47.0,
                53.0,
                55.0,
            ]
        )

        gripper_knots = np.array(
            [
                0.02,  # open
                0.02,
                0.00,  # close on right door handle
                0.00,
                0.02,  # release right door
                0.02,
                0.00,  # close on left door handle
                0.00,
                0.055,  # release left door, open wide
                0.055,  # stay open during descent
                0.00,  # close at the grasp point, over 3 s (44 -> 47)
                0.00,  # stay closed through the lift, move to shelf (47 -> 53)
                0.055,  # open wide after placing, return with open gripper
            ]
        ).reshape(1, 13)
        g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)

        simulator, station_plant = build_and_simulate_trajectory(
            meshcat, q_traj, g_traj, 55.0
        )


if __name__ == "__main__":
    main()
