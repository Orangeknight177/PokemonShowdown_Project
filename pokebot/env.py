"""Environment: poke-env SinglesEnv wired to our embedding and reward.

poke-env (>= 0.9) replaced the old `Gen9EnvSinglePlayer` class with `PokeEnv` /
`SinglesEnv`, a two-agent PettingZoo environment. This module provides:

    PokeRLEnv                a SinglesEnv subclass with our embed_battle / calc_reward
    make_baseline_env()      single-agent Gymnasium env: agent vs a scripted opponent
    make_selfplay_vec_env()  vectorised env where BOTH sides are the learning agent

Observations are dicts: {"observation": float32[N_FEATURES], "action_mask": int8[26]}
(26 actions in gen 9: 6 switches + 4 moves x (1 normal + 4 gimmick variants)).
The mask is produced by SinglesEnv and marks which actions are legal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration

from pokebot.config import DEFAULT_BATTLE_FORMAT, make_server_configuration
from pokebot.features import N_FEATURES, embed_battle
from pokebot.rewards import RewardConfig, RewardTracker

# Scripted opponents usable for training and evaluation.
BASELINE_OPPONENTS = {
    "random": RandomPlayer,
    "maxpower": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


class PokeRLEnv(SinglesEnv):
    """SinglesEnv with this project's observation embedding and reward."""

    def __init__(self, *, reward_config: Optional[RewardConfig] = None, **kwargs):
        super().__init__(**kwargs)
        # PokeEnv wraps this Box into Dict(observation=Box, action_mask=Box).
        self.observation_spaces = {
            agent: Box(-1.0, 1.0, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.reward_tracker = RewardTracker(reward_config)

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_tracker(battle)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return embed_battle(battle)


def make_baseline_env(
    opponent: str = "heuristic",
    *,
    battle_format: str = DEFAULT_BATTLE_FORMAT,
    server_configuration: Optional[ServerConfiguration] = None,
    reward_config: Optional[RewardConfig] = None,
    log_level: int = 40,
    monitor: bool = True,
):
    """Single-agent env: the learner plays against a scripted `opponent`.

    `monitor=True` wraps the env in stable-baselines3's Monitor so episode
    reward / length show up in the training logs (imports SB3 lazily).
    """
    if opponent not in BASELINE_OPPONENTS:
        raise ValueError(f"unknown opponent {opponent!r}; choose from {sorted(BASELINE_OPPONENTS)}")
    env = PokeRLEnv(
        reward_config=reward_config,
        battle_format=battle_format,
        server_configuration=server_configuration or make_server_configuration(),
        log_level=log_level,
        open_timeout=None,
    )
    # start_listening=False: the scripted opponent is driven by the wrapper
    # directly and never needs its own websocket connection.
    opponent_player = BASELINE_OPPONENTS[opponent](
        battle_format=battle_format, start_listening=False
    )
    wrapped = SingleAgentWrapper(env, opponent_player)
    if monitor:
        from stable_baselines3.common.monitor import Monitor

        wrapped = Monitor(wrapped)
    return wrapped


def make_selfplay_vec_env(
    num_envs: int,
    *,
    battle_format: str = DEFAULT_BATTLE_FORMAT,
    server_configuration: Optional[ServerConfiguration] = None,
    reward_config: Optional[RewardConfig] = None,
    log_level: int = 40,
):
    """Self-play: both sides of every battle are controlled by the same policy.

    SuperSuit turns the two-agent PettingZoo env into a vectorised env with one
    sub-environment per agent, so `num_envs` battles give `2 * num_envs` parallel
    sub-environments and the learner trains on both perspectives of each game.
    """
    import supersuit as ss  # optional dependency, only needed for self-play

    env = PokeRLEnv(
        reward_config=reward_config,
        battle_format=battle_format,
        server_configuration=server_configuration or make_server_configuration(),
        log_level=log_level,
        open_timeout=None,
    )
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    return ss.concat_vec_envs_v1(
        vec_env,
        num_vec_envs=num_envs,
        num_cpus=num_envs,
        base_class="stable_baselines3",
    )
