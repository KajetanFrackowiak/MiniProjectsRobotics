"""Base environment configuration shared by all ANYmal expert tasks."""

from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    generated_commands,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANYMAL_XML = PROJECT_ROOT / "assets" / "anybotics_anymal_c" / "anymal_c.xml"

_JOINT_VEL_LIMIT = 12.0   # rad/s
_TORQUE_LIMIT = 80.0      # Nm
_FOOT_FORCE_LIMIT = 700.0 # N


def _get_anymal_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ANYMAL_XML))


def get_anymal_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_anymal_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=(".*")),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.56),
            joint_pos={
                ".*HAA": float(np.radians(0.0)),
                ".*HFE": float(np.radians(30.0)),
                ".*KFE": float(np.radians(-45.0)),
            },
            joint_vel={".*": 0.0},
        ),
    )


def make_base_scene(num_envs: int = 1, env_spacing: float = 3.0) -> SceneCfg:
    return SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=None,  # set per-expert (e.g. walk_expert overrides)
        ),
        entities={"robot": get_anymal_entity_cfg()},
        sensors=(
            ContactSensorCfg(
                name="feet_contact",
                primary=ContactMatch(mode="body", pattern=".*SHANK", entity="robot"),
                fields=("found", "force"),
                reduce="netforce",
            ),
            ContactSensorCfg(
                name="collision_contact",
                primary=ContactMatch(mode="body", pattern=".*(THIGH|SHANK)", entity="robot"),
                fields=("found",),
            ),
        ),
        num_envs=num_envs,
        env_spacing=env_spacing,
    )


def make_base_sim() -> SimulationCfg:
    return SimulationCfg(
        mujoco=MujocoCfg(timestep=0.005, iterations=50),
    )


def make_base_observations() -> dict[str, ObservationGroupCfg]:
    terms = {
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=projected_gravity),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel),
        "actions": ObservationTermCfg(func=last_action),
        "velocity_commands": ObservationTermCfg(
            func=generated_commands, params={"command_name": "base_velocity"}
        ),
    }
    return {
        "policy": ObservationGroupCfg(terms),
    }


def make_base_actions() -> dict[str, ActionTermCfg]:
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*"),
            scale=0.5,
            use_default_offset=True,
        )
    }


def make_base_commands() -> dict[str, CommandTermCfg]:
    return {
        "base_velocity": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
    }


class AnymalBaseEnvCfg(ManagerBasedRlEnvCfg):
    """Base environment configuration for ANYmal expert environments.

    Subclasses should call ``super().__init__(num_envs=..., episode_length_s=...)
    `` and then override scene / rewards / commands as needed.
    """

    def __init__(self, num_envs: int = 1, episode_length_s: float = 10.0):
        super().__init__(
            decimation=4,
            scene=make_base_scene(num_envs=num_envs),
            sim=make_base_sim(),
            observations=make_base_observations(),
            actions=make_base_actions(),
            commands=make_base_commands(),
            rewards={},
            terminations={},
            episode_length_s=episode_length_s,
        )
