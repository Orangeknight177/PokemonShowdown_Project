"""A poke-env Player whose decisions come from any `act(obs, mask) -> action` function.

Keeping this free of torch / stable-baselines3 means the exact same player
class works with a trained network, a random policy (smoke tests) or anything
else that maps (observation, legal-action mask) to an action index.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import DefaultBattleOrder, Player

from pokebot.ability_rules import mask_ability_blocked_moves
from pokebot.features import embed_battle

ActFn = Callable[[np.ndarray, np.ndarray], int]


class PolicyPlayer(Player):
    """Plays by embedding each battle state exactly like the training env does."""

    def __init__(self, act_fn: ActFn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._act_fn = act_fn

    def choose_move(self, battle):
        if battle.wait:  # nothing to decide right now
            return DefaultBattleOrder()
        obs = embed_battle(battle)
        mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        # Prevent known immunity mistakes the trained observation cannot represent.
        action = np.int64(self._act_fn(obs, mask_ability_blocked_moves(battle, mask)))
        # strict=False: if an illegal action ever slips through, fall back to a
        # random legal move instead of crashing a long-running match session.
        return SinglesEnv.action_to_order(action, battle, strict=False)


def random_act_fn(seed: Optional[int] = None) -> ActFn:
    """Uniformly random legal action. Useful for smoke tests and as a baseline."""
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, mask: np.ndarray) -> int:
        return int(rng.choice(np.flatnonzero(mask)))

    return act
