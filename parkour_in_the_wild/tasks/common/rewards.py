"""Reward terms shared by all Stage 1 ANYmal expert tasks."""

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs.mdp.rewards import (
    action_rate_l2,
    joint_torques_l2,
    joint_vel_l2,
)
from mjlab.tasks.velocity.mdp import track_angular_velocity, track_linear_velocity

_JOINT_VEL_LIMIT = 12.0   # rad/s
_TORQUE_LIMIT = 80.0      # Nm
_FOOT_FORCE_LIMIT = 700.0 # N


# ---------------------------------------------------------------------------
# Custom Table 2 reward functions (not available in mjlab's mdp module)
# ---------------------------------------------------------------------------


def joint_velocity_limit(env, asset_cfg=None, limit: float = _JOINT_VEL_LIMIT) -> torch.Tensor:
    """Sum of joint velocities exceeding the limit: Σ max(|q̇ᵢ| − q̇_lim, 0)."""
    asset = env.scene["robot"]
    qvel = asset.data.joint_vel
    return torch.sum(torch.clamp(torch.abs(qvel) - limit, min=0.0), dim=1)


def torque_limit(env, asset_cfg=None, limit: float = _TORQUE_LIMIT) -> torch.Tensor:
    """Sum of actuator torques exceeding the limit: Σ max(|τᵢ| − τ_lim, 0)."""
    asset = env.scene["robot"]
    torque = asset.data.actuator_force
    return torch.sum(torch.clamp(torch.abs(torque) - limit, min=0.0), dim=1)


def base_acceleration(env, asset_cfg=None) -> torch.Tensor:
    """Base accel proxy: ||v̇||² + 0.02||ω̇||² using joint accel magnitude."""
    asset = env.scene["robot"]
    joint_acc = asset.data.joint_acc
    lin_acc = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
    ang_acc = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
    return lin_acc + 0.02 * ang_acc + 0.01 * torch.sum(torch.square(joint_acc), dim=1)


def feet_acceleration(env, asset_cfg=None) -> torch.Tensor:
    """Sum of feet linear acceleration magnitudes: Σ_f ||v̇_f|| (joint accel proxy)."""
    asset = env.scene["robot"]
    joint_acc = asset.data.joint_acc
    return torch.sum(torch.abs(joint_acc), dim=1)


def feet_contact_force(env, sensor_name: str = "feet_contact", limit: float = _FOOT_FORCE_LIMIT) -> torch.Tensor:
    """Penalize feet contact force exceeding limit: Σ max(||F_f|| − 700, 0)²."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_norm = torch.norm(data.force, dim=-1)  # [B, N]
    excess = torch.clamp(force_norm - limit, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def dont_wait(env, asset_cfg=None, threshold: float = 0.2) -> torch.Tensor:
    """Penalize standing still: I(||v_b|| < 0.2)."""
    asset = env.scene["robot"]
    speed = torch.norm(asset.data.root_link_lin_vel_b, dim=-1)
    return (speed < threshold).float()


def stand_at_target(env, asset_cfg=None) -> torch.Tensor:
    """Penalize joint deviation from default when at target: S_L ||q − q_d||."""
    asset = env.scene["robot"]
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    return torch.sum(torch.abs(joint_pos - default_pos), dim=1)


def undesired_collisions(env, sensor_name: str = "collision_contact") -> torch.Tensor:
    """Penalize knee/shank collisions: I_{collision}."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    if data.found is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (data.found > 0).any(dim=1).float()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def get_base_rewards() -> dict[str, RewardTermCfg]:
    """Returns standard reward terms shared across Stage 1 experts."""
    return {
        # --- Command Tracking Rewards ---
        "track_lin_vel": RewardTermCfg(
            func=track_linear_velocity,
            weight=1.5,
            params={"command_name": "base_velocity", "std": 0.25},
        ),
        "track_ang_vel": RewardTermCfg(
            func=track_angular_velocity,
            weight=0.75,
            params={"command_name": "base_velocity", "std": 0.5},
        ),

        # --- Smoothness & Regularization Penalties ---
        "joint_vel": RewardTermCfg(func=joint_vel_l2, weight=-1e-3),
        "torque": RewardTermCfg(func=joint_torques_l2, weight=-1e-5),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-1e-2),
        "base_acc": RewardTermCfg(func=base_acceleration, weight=-1e-3),
        "feet_acc": RewardTermCfg(func=feet_acceleration, weight=-2e-3),

        # --- Actuator & Safety Limits ---
        "joint_vel_limit": RewardTermCfg(
            func=joint_velocity_limit,
            weight=-1.0,
            params={"limit": _JOINT_VEL_LIMIT},
        ),
        "torque_limit": RewardTermCfg(
            func=torque_limit,
            weight=-0.2,
            params={"limit": _TORQUE_LIMIT},
        ),
        "feet_force": RewardTermCfg(
            func=feet_contact_force,
            weight=-1e-5,
            params={"sensor_name": "feet_contact", "limit": _FOOT_FORCE_LIMIT},
        ),
        "collision": RewardTermCfg(
            func=undesired_collisions,
            weight=-1.0,
            params={"sensor_name": "collision_contact"},
        ),

        # --- Task Logic ---
        "dont_wait": RewardTermCfg(
            func=dont_wait,
            weight=-1.0,
            params={"threshold": 0.2},
        ),
        "stand_at_target": RewardTermCfg(
            func=stand_at_target,
            weight=-0.5,
        ),
    }
