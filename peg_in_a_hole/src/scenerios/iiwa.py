from typing import Literal

from utils.assets import (
    HOLE_SDF_PATH,
    IIWA_URDF_PATH,
    PEG_URDF_PATH,
    TABLE_SDF_PATH,
)
from utils.trajectory import get_peg_frame_offset

IIWA_URDF_URI = f"file://{IIWA_URDF_PATH}"
TABLE_SDF_URI = f"file://{TABLE_SDF_PATH}"
PEG_URDF_URI = f"file://{PEG_URDF_PATH}"
HOLE_SDF_URI = f"file://{HOLE_SDF_PATH}"


def make_scenario_string(
    peg_frame_placement: Literal["back", "center", "tip"] = "center",
) -> str:
    peg_frame_offset = get_peg_frame_offset(peg_frame_placement)
    scenario_string = f"""
directives:
- add_model:
    name: iiwa
    file: {IIWA_URDF_URI}
    default_joint_positions:
        iiwa_joint_2: [0.1]
        iiwa_joint_4: [-1.2]
        iiwa_joint_6: [1.6]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: robot_table
    file: {TABLE_SDF_URI}
- add_weld:
    parent: world
    child: robot_table::link
    X_PC:
        translation: [0, 0, -0.7645]
- add_model:
    name: work_table
    file: {TABLE_SDF_URI}
- add_weld:
    parent: world
    child: work_table::link
    X_PC:
        translation: [0.75, 0, -0.7645]
- add_model:
    name: peg
    file: {PEG_URDF_URI}
- add_weld:
    parent: iiwa::iiwa_link_7
    child: peg::peg_body_link
    X_PC:
        translation: [0, 0, 0.10]
        rotation: !Rpy {{ deg: [0, 180, -90] }}
- add_frame:
    name: peg_tip_frame
    X_PF:
      base_frame: peg::peg_body_link
      translation: [0, 0, {peg_frame_offset}]
      rotation: !Rpy {{ deg: [0, 0, 0] }}
- add_model:
    name: hole
    file: {HOLE_SDF_URI}
- add_weld:
    parent: work_table::link
    child: hole::hole_chamfered_body_link
    X_PC:
        translation: [0, 0.1, 0.7645]
        rotation: !Rpy {{ deg: [0, 0, 0] }}
model_drivers:
    iiwa: !IiwaDriver
      control_mode: torque_only
      desired_kp_gains: [500, 500, 200]
      hand_model_name: wsg
  """
    return scenario_string
