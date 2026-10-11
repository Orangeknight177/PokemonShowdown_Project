"""pokebot - a self-improving Pokemon Showdown bot built on poke-env + stable-baselines3.

Module map (only `policy` needs torch / stable-baselines3):

    config    defaults and server-configuration helper
    features  embed_battle(): battle state -> fixed-size float vector
    rewards   RewardConfig + RewardTracker: potential-based reward shaping
    env       PokeRLEnv (poke-env SinglesEnv subclass) and env factories
    players   PolicyPlayer: a poke-env Player driven by any act(obs, mask) function
    policy    action-masked actor-critic policy and checkpoint loading
"""
