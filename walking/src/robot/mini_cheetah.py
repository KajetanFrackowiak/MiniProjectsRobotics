from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class MiniCheetah:

    URDF_PATH = ROOT / "assets" / "mini_cheetah_urdf" / "urdf" / "mini_cheetah.urdf"
    BASE_BODY_NAME = "body"
    # Toe collision sphere radius (from the URDF). Contact happens at the
    # sphere surface, not at the toe frame, so it must be accounted for.
    TOE_RADIUS = 0.015
    # Height of the base origin when standing = leg extension (hip to toe
    # center, 0.27 m with Q_NOMINAL) + toe radius.
    BASE_HEIGHT = 0.27 + TOE_RADIUS

    # << = wszystkie kolana do tyłu (jak Spot / Go1 / ANYmal) - jedyny
    # wspierany wariant (Bledt 2018 / Kim 2019). FK: leg extension 0.27m,
    # toe radius 0.015m => BASE_HEIGHT 0.285m. Przednie stopy w << są ~4cm
    # cofnięte vs <> (x=0.169 vs 0.211) ale ta sama wysokość. Kolizja
    # body-thigh dla << (~17mm) jest filtrowana w simulation.py.
    Q_NOMINAL = np.array([
        0.0, -0.804,  1.607,   # FR
        0.0, -0.804,  1.607,   # FL
        0.0, -0.804,  1.607,   # HR
        0.0, -0.804,  1.607,   # HL
    ])
    # Słownik konfiguracji kolan: literalnie tylko "<<"
    KNEE_CONFIGS = {
        "<<": Q_NOMINAL,
    }

    # 2. Skaluj wzmocnienia KP i KD pod dynamikę chodu
    KP = np.array([60.0] * 12)
    KD = np.array([3.5] * 12)
    MAX_TORQUE = 35.0

    JOINT_NAMES = [
        "torso_to_abduct_fr_j",
        "abduct_fr_to_thigh_fr_j",
        "thigh_fr_to_knee_fr_j",

        "torso_to_abduct_fl_j",
        "abduct_fl_to_thigh_fl_j",
        "thigh_fl_to_knee_fl_j",

        "torso_to_abduct_hr_j",
        "abduct_hr_to_thigh_hr_j",
        "thigh_hr_to_knee_hr_j",

        "torso_to_abduct_hl_j",
        "abduct_hl_to_thigh_hl_j",
        "thigh_hl_to_knee_hl_j",
    ]

    def __init__(self, plant, model):
        self.plant = plant
        self.model = model

        self.joints = [
            plant.GetJointByName(name, model)
            for name in self.JOINT_NAMES
        ]

    @property
    def num_joints(self):
        return len(self.joints)

    def get_positions(self, plant_context):
        return np.array([
            joint.get_angle(plant_context)
            for joint in self.joints
        ])

    def get_velocities(self, plant_context):
        return np.array([
            joint.get_angular_rate(plant_context)
            for joint in self.joints
        ])

    def set_positions(self, plant_context, positions):
        for joint, pos in zip(self.joints, positions):
            joint.set_angle(plant_context, pos)

    def set_velocities(self, plant_context, velocities):
        for joint, vel in zip(self.joints, velocities):
            joint.set_angular_rate(plant_context, vel)
