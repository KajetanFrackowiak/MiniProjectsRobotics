import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.trajectories import PiecewiseQuaternionSlerp, PiecewisePolynomial

# ---------------------------------------------------------------------------
# Door geometry.
# ---------------------------------------------------------------------------
# Door offsets are expressed in the door-center frame, which shares the
# orientation of the world frame. R = right door, L = left door (mirror of the
# right door about the x-z plane).

p_Wdoor_R = np.array([0.7477, -0.1445, 0.4148])  # center of the right door
p_handle_R = np.array([-0.033, 0.1245, 0])  # handle offset from the door center
p_hinge_R = np.array([0.008, -0.1395, 0])  # hinge offset from the door center

p_Whandle_R = p_Wdoor_R + p_handle_R  # handle position in the world frame
p_Whinge_R = p_Wdoor_R + p_hinge_R  # hinge position in the world frame

p_hinge_to_handle_R = p_handle_R - p_hinge_R
r_hinge_R = np.linalg.norm(p_hinge_to_handle_R)  # distance between hinge and handle
theta_start_R = np.arctan2(p_hinge_to_handle_R[1], p_hinge_to_handle_R[0])
angle_end_R = np.radians(173)  # 180 deg saturates iiwa_joint_6; use 120-160 deg for the easy version

p_Wdoor_L = np.array([0.7477, 0.1445, 0.4148])  # center of the left door
p_handle_L = np.array([-0.033, -0.1245, 0])  # handle offset from the door center
p_hinge_L = np.array([0.008, 0.1395, 0])  # hinge offset from the door center

p_Whandle_L = p_Wdoor_L + p_handle_L  # handle position in the world frame
p_Whinge_L = p_Wdoor_L + p_hinge_L  # hinge position in the world frame

p_hinge_to_handle_L = p_handle_L - p_hinge_L
r_hinge_L = np.linalg.norm(p_hinge_to_handle_L)
theta_start_L = np.arctan2(p_hinge_to_handle_L[1], p_hinge_to_handle_L[0])
angle_end_L = -np.radians(173)

p_retract_G = np.array([0.0, -0.08, 0])  # handle pull-back offset, in the gripper frame

# ---------------------------------------------------------------------------
# Pick target: the sugar box.
# ---------------------------------------------------------------------------

p_sugar = np.array([0.4, 0, 0.215])  # sugar box center (floor object)

# ---------------------------------------------------------------------------
# Pose helpers.
# ---------------------------------------------------------------------------


def evaluate_pose(rot_traj, pos_traj, t: float) -> RigidTransform:
    return RigidTransform(
        RotationMatrix(rot_traj.orientation(t)),
        pos_traj.value(t),
    )


def _open_door_pose(
    t: float,
    p_Whinge: np.ndarray,
    r_hinge: float,
    theta_start: float,
    angle_end: float,
    yaw_offset: float,
) -> RigidTransform:
    # Interpolate the handle angle about the hinge, then place the gripper on it.
    theta = theta_start + (angle_end - theta_start) * t
    p_Whandle = r_hinge * np.array([np.cos(theta), np.sin(theta), 0.0]) + p_Whinge
    R_Whandle = RollPitchYaw(0, 0, theta + yaw_offset).ToRotationMatrix()
    X_Whandle = RigidTransform(R_Whandle, p_Whandle)
    # Add a small offset to account for the gripper.
    p_handleG = np.array([0.0, 0.1, 0.0])
    R_handleG = RollPitchYaw(0, np.pi, np.pi).ToRotationMatrix()
    X_handleG = RigidTransform(R_handleG, p_handleG)
    return X_Whandle.multiply(X_handleG)


def interpolate_open_pose(t: float) -> RigidTransform:
    # Gripper pose while sweeping the right door open (t in [0, 1]).
    return _open_door_pose(t, p_Whinge_R, r_hinge_R, theta_start_R, angle_end_R, 0.0)


def interpolate_open_left_pose(t: float) -> RigidTransform:
    # Gripper pose while sweeping the left door open (t in [0, 1]).
    # +pi yaw: grab the handle from the front.
    return _open_door_pose(t, p_Whinge_L, r_hinge_L, theta_start_L, angle_end_L, np.pi)

# ---------------------------------------------------------------------------
# Phase 1: entry - approach the right-door handle.
# ---------------------------------------------------------------------------


def make_entry_orientation_trajectory(start_pose) -> PiecewiseQuaternionSlerp:
    traj = PiecewiseQuaternionSlerp()
    R_start = start_pose.rotation()
    R_handle = interpolate_open_pose(0.0).rotation()
    traj.Append(0.0, R_start)
    traj.Append(2.5, R_start)
    traj.Append(5.0, R_handle)
    return traj


def make_entry_position_trajectory(start_pose) -> PiecewisePolynomial:
    p_start = start_pose.translation()
    p_handle = interpolate_open_pose(0.0).translation()
    p_mid = np.array([
        (p_start[0] + p_handle[0]) / 2.0,
        (p_start[1] + p_handle[1]) / 2.0,
        0.5,
    ])
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 2.5, 5.0],
        np.column_stack([p_start, p_mid, p_handle]),
    )

# ---------------------------------------------------------------------------
# Phase 2: retract from the right door.
# ---------------------------------------------------------------------------


def make_retract_right_position_trajectory() -> PiecewisePolynomial:
    X_open = interpolate_open_pose(1.0)
    p0 = X_open.translation()
    # Pull back from the handle (offset in the gripper frame).
    p1 = p0 + X_open.rotation() @ p_retract_G
    # Continue retracting in the world frame.
    p2 = p1 + np.array([-0.1, 0, 0])
    p3 = p2 + np.array([-0.05, 0, 0])
    p4 = p3 + np.array([0.0, 0.20, 0])
    p5 = p4 + np.array([0.05, 0.25, 0])
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        np.column_stack([p0, p1, p2, p3, p4, p5]),
    )


def make_retract_right_orientation_trajectory() -> PiecewiseQuaternionSlerp:
    traj = PiecewiseQuaternionSlerp()
    R = interpolate_open_pose(1.0).rotation()
    R_tilt = R @ RollPitchYaw(0, 0, np.radians(40)).ToRotationMatrix()
    traj.Append(0.0, R)
    traj.Append(2.0, R)
    traj.Append(4.0, R_tilt)
    traj.Append(6.0, R_tilt)
    traj.Append(8.0, R)
    traj.Append(10.0, R)
    return traj

# ---------------------------------------------------------------------------
# Phase 3: entry to the left-door handle.
# ---------------------------------------------------------------------------


def make_entry_left_orientation_trajectory(start_pose) -> PiecewiseQuaternionSlerp:
    traj = PiecewiseQuaternionSlerp()
    R_start = start_pose.rotation()
    R_handle = interpolate_open_left_pose(0.0).rotation()
    traj.Append(0.0, R_start)
    traj.Append(3.0, R_start)
    traj.Append(4.0, R_handle)
    traj.Append(5.0, R_handle)
    return traj


def make_entry_left_position_trajectory(start_pose) -> PiecewisePolynomial:
    p_start = start_pose.translation()
    p_handle = interpolate_open_left_pose(0.0).translation()
    p_mid = np.array([0.58, p_start[1], 0.55])
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 3.0, 4.0, 5.0],
        np.column_stack([p_start, p_mid, p_mid, p_handle]),
    )

# ---------------------------------------------------------------------------
# Phase 4: retract from the left door (ends above the sugar box).
# ---------------------------------------------------------------------------


def make_retract_left_position_trajectory() -> PiecewisePolynomial:
    X_open = interpolate_open_left_pose(1.0)
    p0 = X_open.translation()
    # Pull back from the handle (offset in the gripper frame).
    p1 = p0 + X_open.rotation() @ p_retract_G
    # Continue retracting in the world frame.
    p2 = p1 + np.array([-0.1, 0, 0])
    # End above the sugar box.
    p3 = np.array([p_sugar[0], p_sugar[1], p0[2]])
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 2.0, 4.0, 6.0],
        np.column_stack([p0, p1, p2, p3]),
    )


def make_retract_left_orientation_trajectory() -> PiecewiseQuaternionSlerp:
    traj = PiecewiseQuaternionSlerp()
    R = interpolate_open_left_pose(1.0).rotation()
    traj.Append(0.0, R)
    traj.Append(6.0, R)
    return traj

# ---------------------------------------------------------------------------
# Phase 5: pick the sugar box, place it on the shelf, and return.
# ---------------------------------------------------------------------------


def make_pick_position_trajectory() -> PiecewisePolynomial:
    p_above = np.array([p_sugar[0], p_sugar[1], 0.4148])
    p_grasp = np.array([p_above[0], p_above[1], 0.15])  # top of the sugar box
    p_lift = np.array([p_sugar[0], p_sugar[1], 0.65])
    p_rotate = np.array([0.5, p_lift[1] + 0.1, 0.65])
    p_place = np.array([0.91, p_rotate[1], p_rotate[2]])
    p_return = np.array([p_place[0] - 0.4, p_place[1], p_place[2]])
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 2.0, 4.0, 7.0, 9.0, 10.0, 11.0, 13.0, 15.0],
        np.column_stack([p_above, p_above, p_grasp, p_grasp, p_lift, p_rotate, p_rotate, p_place, p_return]),
    )


def make_pick_orientation_trajectory() -> PiecewiseQuaternionSlerp:
    traj = PiecewiseQuaternionSlerp()
    R_start = interpolate_open_left_pose(1.0).rotation()
    R_pick = RotationMatrix(RollPitchYaw(np.radians(-90), 0, 0))
    R_place = R_pick @ RollPitchYaw(0, 0, np.radians(-90)).ToRotationMatrix()
    traj.Append(0.0, R_start)
    traj.Append(2.0, R_pick)
    traj.Append(10.0, R_pick)
    traj.Append(11.0, R_place)
    traj.Append(15.0, R_place)
    return traj

# ---------------------------------------------------------------------------
# Master dispatch: end-effector pose over the full 55 s task.
# ---------------------------------------------------------------------------


def interpolate_pose(
    t: float,
    entry_rot: PiecewiseQuaternionSlerp,
    entry_pos: PiecewisePolynomial,
    retract_right_rot: PiecewiseQuaternionSlerp,
    retract_right_pos: PiecewisePolynomial,
    entry_left_rot: PiecewiseQuaternionSlerp,
    entry_left_pos: PiecewisePolynomial,
    retract_left_rot: PiecewiseQuaternionSlerp,
    retract_left_pos: PiecewisePolynomial,
    pick_rot: PiecewiseQuaternionSlerp,
    pick_pos: PiecewisePolynomial,
) -> RigidTransform:
    if t < 5.0:
        # entry: approach the right-door handle
        return evaluate_pose(entry_rot, entry_pos, t)

    if t < 6.0:
        # wait at the handle
        return evaluate_pose(entry_rot, entry_pos, 5.0)

    if t < 11.0:
        # open the right door
        return interpolate_open_pose((t - 6.0) / 5.0)

    if t < 21.0:
        # retract after releasing the right door
        return evaluate_pose(retract_right_rot, retract_right_pos, t - 11.0)

    if t < 26.0:
        # entry to the left-door handle (rotate at the waypoint, then slide in)
        return evaluate_pose(entry_left_rot, entry_left_pos, t - 21.0)

    if t < 27.0:
        # wait at the handle
        return evaluate_pose(entry_left_rot, entry_left_pos, 5.0)

    if t < 32.0:
        # open the left door
        return interpolate_open_left_pose((t - 27.0) / 5.0)

    if t < 34.0:
        # hold the left door open while releasing the handle
        return interpolate_open_left_pose(1.0)

    if t < 40.0:
        # retract away from the opened doors, ending above the sugar box
        return evaluate_pose(retract_left_rot, retract_left_pos, t - 34.0)

    if t < 55.0:
        # pick the sugar box, place it on the shelf, and return
        return evaluate_pose(pick_rot, pick_pos, t - 40.0)

    # hold the final pose
    return evaluate_pose(pick_rot, pick_pos, 15.0)
