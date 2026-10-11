#!/usr/bin/env python3
"""Sanity checks that need a running Showdown server but NO torch / SB3.

    python smoke_test.py --battles 3

What it verifies
  1. Observations have the right shape, dtype and range at every step, and the
     action mask always contains at least one legal action.
  2. The per-step rewards telescope: their sum equals the value of the final
     battle state (see pokebot/rewards.py).
  3. PolicyPlayer (the class used for evaluation / live play) can finish
     battles driven by a random legal-action policy.
Run this once after installing, before spending hours on training.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
from poke_env.player import RandomPlayer

from pokebot.config import DEFAULT_BATTLE_FORMAT, DEFAULT_HOST, DEFAULT_PORT, make_server_configuration
from pokebot.env import make_baseline_env
from pokebot.features import FEATURE_NAMES, N_FEATURES
from pokebot.players import PolicyPlayer, random_act_fn
from pokebot.rewards import RewardConfig, state_value


def check_env(n_battles: int, battle_format: str, server) -> None:
    cfg = RewardConfig()
    env = make_baseline_env(
        "heuristic",
        battle_format=battle_format,
        server_configuration=server,
        reward_config=cfg,
        monitor=False,
    )
    rng = np.random.default_rng(0)
    try:
        for battle_idx in range(n_battles):
            obs, _ = env.reset()
            total_reward, steps = 0.0, 0
            while True:
                vec, mask = obs["observation"], obs["action_mask"]
                assert vec.shape == (N_FEATURES,) and vec.dtype == np.float32, (vec.shape, vec.dtype)
                assert np.isfinite(vec).all(), "NaN/inf in observation"
                assert vec.min() >= -1.0 and vec.max() <= 1.0, "observation outside [-1, 1]"
                assert mask.sum() >= 1, "no legal action"
                action = np.int64(rng.choice(np.flatnonzero(mask)))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            battle = env.env.battle1
            final_value = state_value(battle, cfg)
            assert battle.finished
            assert abs(total_reward - final_value) < 1e-6, (total_reward, final_value)
            print(
                f"[env] battle {battle_idx + 1}: {'won' if battle.won else 'lost'} in {steps} "
                f"decisions, return {total_reward:+.2f} == final state value {final_value:+.2f}"
            )
    finally:
        env.close()


async def check_player(n_battles: int, battle_format: str, server) -> None:
    agent = PolicyPlayer(
        random_act_fn(1),
        battle_format=battle_format,
        server_configuration=server,
        max_concurrent_battles=n_battles,
    )
    opponent = RandomPlayer(
        battle_format=battle_format, server_configuration=server, max_concurrent_battles=n_battles
    )
    await agent.battle_against(opponent, n_battles=n_battles)
    assert agent.n_finished_battles == n_battles
    print(f"[player] finished {agent.n_finished_battles} battles, won {agent.n_won_battles}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--battle-format", default=DEFAULT_BATTLE_FORMAT)
    parser.add_argument("--battles", type=int, default=3)
    args = parser.parse_args()

    server = make_server_configuration(args.host, args.port)
    print(f"observation size: {N_FEATURES} features (first 3: {FEATURE_NAMES[:3]})")
    check_env(args.battles, args.battle_format, server)
    asyncio.run(check_player(args.battles, args.battle_format, server))
    print("all checks passed")


if __name__ == "__main__":
    main()
