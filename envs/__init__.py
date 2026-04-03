from gymnasium.envs.registration import register

register(
    id="ReducedBankedGliderPullout-v0",
    entry_point="envs.reduced_banked_pullout:ReducedBankedGliderPullout",
    max_episode_steps=100,
)