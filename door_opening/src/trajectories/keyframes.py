import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from stations.station import setup_manipulation_station
from trajectories.trajectory import (
    interpolate_pose,
    make_entry_orientation_trajectory,
    make_entry_position_trajectory,
    make_retract_right_position_trajectory,
    make_retract_right_orientation_trajectory,
    make_entry_left_orientation_trajectory,
    make_entry_left_position_trajectory,
    make_retract_left_position_trajectory,
    make_retract_left_orientation_trajectory,
    make_pick_position_trajectory,
    make_pick_orientation_trajectory,
)
from manipulation.meshcat_utils import AddMeshcatTriad

def add_poses(meshcat, t_lst, show_triads=True):
    initial_pose = setup_manipulation_station(meshcat)

    entry_rot = make_entry_orientation_trajectory(initial_pose)
    entry_pos = make_entry_position_trajectory(initial_pose)

    retract_right_rot = make_retract_right_orientation_trajectory()
    retract_right_pos = make_retract_right_position_trajectory()

    retract_end = RigidTransform(
        RotationMatrix(retract_right_rot.orientation(10.0)),
        retract_right_pos.value(10.0),
    )
    entry_left_rot = make_entry_left_orientation_trajectory(retract_end)
    entry_left_pos = make_entry_left_position_trajectory(retract_end)

    retract_left_rot = make_retract_left_orientation_trajectory()
    retract_left_pos = make_retract_left_position_trajectory()

    pick_rot = make_pick_orientation_trajectory()
    pick_pos = make_pick_position_trajectory()


    pose_lst = []
    for t in t_lst:
        X_WG = interpolate_pose(
            t,
            entry_rot,
            entry_pos,
            retract_right_rot,
            retract_right_pos,
            entry_left_rot,
            entry_left_pos,
            retract_left_rot,
            retract_left_pos,
            pick_rot,
            pick_pos,
        )

        if show_triads:
            AddMeshcatTriad(
                meshcat,
                path=str(t),
                X_PT=X_WG,
                opacity=0.2,
            )

        pose_lst.append(X_WG)

    return pose_lst
