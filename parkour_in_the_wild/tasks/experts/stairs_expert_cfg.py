from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainGeneratorCfg
import mjlab.terrains as terrain_gen

from tasks.common.base_env_cfg import AnymalBaseEnvCfg
from tasks.common.rewards import get_base_rewards


class StartsExpertEnvCfg(AnymalBaseEnvCfg):
    """Configuration for vertical climbing and platform scaling"""

    def __init__(self, num_envs: int = 1, episode_length_s: float = 10.0):
        super().__init__(num_envs=num_envs, episode_length_s=episode_length_s)

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=5,
            border_width=20.0,
            curriculum=True,
            sub_terrains={
                "pyramid_starts": terrain_gen.BoxPyramidStairsTerrainCfg(
                    proportion=0.5,
                    step_height_range=(0.05, 0.20),
                    step_width=0.35,
                    platform_width=2.0,
                ),
                "grid_platforms": terrain_gen.BoxRandomGridTerrainCfg(
                    proportion=0.5,
                    grid_width=0.4,
                    grid_height_range=(0.05, 0.25),
                    platform_width=1.5,
                ),
            },
        )

        self.rewards = get_base_rewards()
        # Relax base acceleration penalty slightly to allow pitching/climbing
        self.rewards["base_acc"].weight = -5e-4


def stairs_expert_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """RL runner config for the climb/stairs expert"""
    return RslRlOnPolicyRunnerCfg(
        experiment_name="anymal_stairs_expert"
    )


register_mjlab_task(
    task_id="Mjlab-Anymal-Stairs-v0",
    env_cfg=StartsExpertEnvCfg(),
    play_env_cfg=StartsExpertEnvCfg(episode_length_s=1e10),
    rl_cfg=stairs_expert_rl_cfg(),
)