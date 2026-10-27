from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainGeneratorCfg
import mjlab.terrains as terrain_gen

from tasks.common.base_env_cfg import AnymalBaseEnvCfg
from tasks.common.rewards import get_base_rewards

class StonesExpertEnvCfg(AnymalBaseEnvCfg):
    """Configuration for discrete stepping stones and pit leaping"""

    def __init__(self, num_envs: int = 1, episode_length_s: float = 10.0):
        super().__init__(num_envs=num_envs, episode_length_s=episode_length_s)

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=5,
            border_width=20.0,
            curriculum=True,
            sub_terrains={
                "stepping_stones": terrain_gen.BoxSteppingStonesTerrainCfg(
                    proportion=1.0,
                    stone_size_range=(0.4, 0.8),
                    stone_distance_range=(0.05, 0.25),
                    stone_height=0.15,
                    floor_depth=0.5,
                ),
            },
        )

        self.rewards = get_base_rewards()
        # Slightly reduce feet acceleration penalty to allow quick leg swings over gaps
        self.rewards["feet_acc"].weight = -1e-3


def stones_expert_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """RL runner config for the stepping stones expert"""
    return RslRlOnPolicyRunnerCfg(
        experiment_name="anymal_stones_expert",
    )

register_mjlab_task(
    task_id="Mjlab-Anymal-Stones-v0",
    env_cfg=StonesExpertEnvCfg(),
    play_env_cfg=StonesExpertEnvCfg(episode_length_s=1e10),
    rl_cfg=stones_expert_rl_cfg(),
)