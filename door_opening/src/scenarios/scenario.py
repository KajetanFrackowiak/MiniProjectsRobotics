from manipulation.station import LoadScenario, Scenario

from utils.asset_paths import (
    AMAZON_TABLE_SIMPLIFIED_URI,
    CAMERA_BOX_URI,
    CUPBOARD_URI,
    IIWA_SDF_7_NO_COLLISION_URI,
    IIWA_SDF_7_WITH_BOX_COLLISION_URI,
    SCHUNK_WSG_50_TIP_URI,
    SUGAR_BOX_URI,
)


def load_iiwa_scenario(iiwa_collision: bool = False) -> Scenario:
    """Load the iiwa scenario YAML into a typed Scenario object."""
    return LoadScenario(data=make_iiwa_scenario(iiwa_collision=iiwa_collision))


def make_iiwa_scenario(iiwa_collision: bool = False) -> str:
    if iiwa_collision:
        iiwa_file = IIWA_SDF_7_WITH_BOX_COLLISION_URI
    else:
        iiwa_file = IIWA_SDF_7_NO_COLLISION_URI

    robot_scenario = f"""
        directives:
        - add_model:
            name: iiwa
            file: {iiwa_file}
            default_joint_positions:
                iiwa_joint_1: [0]
                iiwa_joint_2: [0.6]
                iiwa_joint_3: [0]
                iiwa_joint_4: [-1.75]
                iiwa_joint_5: [0]
                iiwa_joint_6: [ 1.0]
                iiwa_joint_7: [0]
        - add_weld:
            parent: world
            child: iiwa::iiwa_link_0
        - add_model:
            name: wsg
            file: {SCHUNK_WSG_50_TIP_URI}
        - add_weld:
            parent: iiwa::iiwa_link_7
            child: wsg::body
            X_PC:
                translation: [0, 0, 0.114]
                rotation: !Rpy {{ deg: [90, 0, 90] }}
        - add_model:
            name: table
            file: {AMAZON_TABLE_SIMPLIFIED_URI}
        - add_weld:
            parent: world
            child: table::amazon_table
            X_PC:
                translation: [0.3257, 0, -0.0127]
        - add_model:
            name: cupboard
            file: {CUPBOARD_URI}
        - add_weld:
            parent: world
            child: cupboard::cupboard_body
            X_PC:
                translation: [0.9057, 0, 0.4148]
                rotation: !Rpy {{ deg: [0, 0, 180] }}
        - add_model:
            name: sugar_floor
            file: {SUGAR_BOX_URI}
            default_free_body_pose:
                base_link_sugar:
                    base_frame: world
                    translation: [0.4, 0, 0.215]
                    rotation: !Rpy {{ deg: [0, 90, 0] }}
        - add_model:
            name: sugar_shelf_up
            file: {SUGAR_BOX_URI}
            default_free_body_pose:
                base_link_sugar:
                    base_frame: world
                    translation: [0.91, -0.2, 0.65]
                    rotation: !Rpy {{ deg: [-90, 0, -90] }}
        - add_model:
            name: camera0
            file: {CAMERA_BOX_URI}
        - add_weld:
            parent: world
            child: camera0::base
            X_PC:
                translation: [-0.228895, -0.452176, 0.486308]
                rotation: !Rpy {{ deg: [146.0, 78.0, 170] }}
        - add_model:
            name: camera1
            file: {CAMERA_BOX_URI}
        - add_weld:
            parent: world
            child: camera1::base
            X_PC:
                translation: [-0.201813, 0.469259, 0.417045]
                rotation: !Rpy {{ deg: [150.0, -76.6, -9.8] }}
        - add_model:
            name: camera2
            file: {CAMERA_BOX_URI}
        - add_weld:
            parent: world
            child: camera2::base
            X_PC:
                translation: [0.786258, -0.048422, 1.043315]
                rotation: !Rpy {{ deg: [150.0, 1.3, 88] }}
        model_drivers:
            iiwa: !IiwaDriver
                control_mode: position_only
                hand_model_name: wsg
            wsg: !SchunkWsgDriver {{}}
        visualization:
            publish_period: 0.1
    """
    return robot_scenario
