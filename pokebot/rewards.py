"""Reward function: potential-based shaping from damage, faints, status and the result.

Idea
    Define a scalar "value" of a battle state from our point of view, then give
    the agent the CHANGE in that value after each turn:

        reward_t = value(state_t) - value(state_{t-1})

    Because the per-step rewards telescope, the total return of an episode is
    simply value(final state) - value(initial state). That makes the shaping
    terms (damage, faints, status) consistent with each other and with the
    final win/loss signal: dealing damage and later healing it back nets zero.

Value of a state (all terms are "from our point of view")

    + damage_dealt  * sum over opponent Pokemon of (1 - hp_fraction)
    - damage_taken  * sum over our Pokemon of (1 - hp_fraction)
    + faint_dealt   * number of fainted opponent Pokemon
    - faint_taken   * number of our fainted Pokemon
    + status_dealt  * number of non-fainted opponent Pokemon with a status
    - status_taken  * number of our non-fainted Pokemon with a status
    + win_bonus   if we won,  - loss_penalty if we lost

Opponent Pokemon we have not seen yet are assumed to be at full health, so they
contribute nothing until revealed.

Worked example with the defaults: knocking out a full-health opponent Pokemon
removes 1.0 HP-fraction and adds one faint, so the reward is 1.0 + 2.0 = 3.0.
Losing one of our own full-health Pokemon is -3.0. Winning adds +30.
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

from poke_env.battle import AbstractBattle


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the reward terms. All values are magnitudes (>= 0)."""

    damage_dealt: float = 1.0  # per full opponent HP bar removed
    damage_taken: float = 1.0  # per full HP bar of ours lost
    faint_dealt: float = 2.0  # per opponent Pokemon knocked out
    faint_taken: float = 2.0  # per Pokemon of ours knocked out
    status_dealt: float = 0.5  # per opponent Pokemon statused
    status_taken: float = 0.5  # per Pokemon of ours statused
    win_bonus: float = 30.0
    loss_penalty: float = 30.0


def state_value(battle: AbstractBattle, cfg: RewardConfig) -> float:
    """Scalar value of the current battle state for the player owning `battle`."""
    own = list(battle.team.values())
    opp = list(battle.opponent_team.values())

    opp_hp_lost = sum(1.0 - m.current_hp_fraction for m in opp)
    own_hp_lost = sum(1.0 - m.current_hp_fraction for m in own)
    opp_fainted = sum(1 for m in opp if m.fainted)
    own_fainted = sum(1 for m in own if m.fainted)
    opp_statused = sum(1 for m in opp if m.status is not None and not m.fainted)
    own_statused = sum(1 for m in own if m.status is not None and not m.fainted)

    value = (
        cfg.damage_dealt * opp_hp_lost
        - cfg.damage_taken * own_hp_lost
        + cfg.faint_dealt * opp_fainted
        - cfg.faint_taken * own_fainted
        + cfg.status_dealt * opp_statused
        - cfg.status_taken * own_statused
    )
    if battle.won:
        value += cfg.win_bonus
    elif battle.lost:
        value -= cfg.loss_penalty
    return value


class RewardTracker:
    """Callable that returns value(now) - value(previous call) for each battle.

    The previous value is stored per battle object (weakly, so finished battles
    are garbage collected). A fresh battle starts at value 0 (nothing has
    happened yet). The tracker pickles to just its config, which keeps the
    environment picklable for SubprocVecEnv / SuperSuit worker processes.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self._last: "WeakKeyDictionary[AbstractBattle, float]" = WeakKeyDictionary()

    def __call__(self, battle: AbstractBattle) -> float:
        value = state_value(battle, self.config)
        reward = value - self._last.get(battle, 0.0)
        self._last[battle] = value
        return reward

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        self.config = state["config"]
        self._last = WeakKeyDictionary()
