from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainGeneratorCfg
import mjlab.terrains as terrain_gen

from tasks.common.base_env_cfg import AnymalBaseEnvCfg
from tasks.common.rewards import get_base_rewards

class WallExpertEnvCfg(AnymalBaseEnvCfg):
    """Configuration for vertical hurdle valuting and explsoive jumping"""

    def __init__(self, num_envs: int = 1, episode_length_s: float = 10.0):
        super().__init__(num_envs=num_envs, episode_length_s=episode_length_s)

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=5,
            border_width=20.0,
            curriculum=True,
            sub_terrains={
                "random_hurdles": terrain_gen.BoxRandomSpreadTerrainCfg(
                    proportion=1.0,
                    box_width_range=(0.2, 0.4),
                    box_height_range=(0.15, 0.35),
                    num_boxes=50,
                ),
            },
        )

        self.rewards = get_base_rewards()
        # Relax acceleration and action rate penalties to encourage dynamic launching
        self.rewards["base_acc"].weight = -5e-4
        self.rewards["action_rate"].weight = -5e-3


def wall_expert_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """RL runner config ro the wall/hurdle vaulting expert"""
    return RslRlOnPolicyRunnerCfg(
        experiment_name="anymal_wall_expert",
    )

register_mjlab_task(
    task_id="Mjlab-Anymal-Wall-v0",
    env_cfg=WallExpertEnvCfg(),
    play_env_cfg=WallExpertEnvCfg(episode_length_s=1e10),
    rl_cfg=wall_expert_rl_cfg(),
)

