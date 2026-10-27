from mjlab.terrains import TerrainGeneratorCfg
import mjlab.terrains as terrain_gen
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from tasks.common.base_env_cfg import AnymalBaseEnvCfg
from tasks.common.rewards import get_base_rewards


class WalkExpertEnvCfg(AnymalBaseEnvCfg):
    """Configuration for walking over flat and rough terrains"""
    def __init__(self, num_envs: int = 1, episode_length_s: float = 10.0):
        super().__init__(num_envs=num_envs, episode_length_s=episode_length_s)

        # 50 % flat ground + 50 % random rough noise
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=5,
            border_width=20.0,
            curriculum=True,
            sub_terrains={
                "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.5),
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.5,
                    noise_range=(0.02, 0.08),
                    noise_step=0.01,
                ),
            },
        )

        self.rewards = get_base_rewards()


def walk_expert_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """RL runner config for the walk expert."""
    return RslRlOnPolicyRunnerCfg(
        experiment_name="anymal_walk_expert",
    )


register_mjlab_task(
    task_id="Mjlab-Anymal-Walk-v0",
    env_cfg=WalkExpertEnvCfg(),
    play_env_cfg=WalkExpertEnvCfg(episode_length_s=1e10),
    rl_cfg=walk_expert_rl_cfg(),
)
