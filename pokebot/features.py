"""State embedding: turns a poke-env battle into a fixed-size float vector.

Layout of the vector (see FEATURE_NAMES for the exact order):

    own active Pokemon     hp fraction, 7 stat boosts, 6 status flags
    opponent active        hp fraction, 7 stat boosts, 6 status flags
    matchup                opponent_threat, speed_advantage, can_tera
    4 move slots           base_power, effectiveness, stab, accuracy,
                           damage_score, pp_fraction, available
    6 team slots           hp fraction, best_damage vs the opponent's active
    team counts            own_remaining, opp_remaining

Every feature is scaled into [-1, 1] (most into [0, 1]) so a plain MLP can
consume the vector without extra normalisation.

IMPORTANT - slot alignment
    SinglesEnv maps action 0-5 to "switch to the i-th Pokemon of battle.team"
    and action 6-9 to "use the i-th *known* move of the active Pokemon". The
    move and team blocks below use exactly the same ordering, so feature block
    i always describes the thing that action i would do. (Indexing by
    `battle.available_moves` instead would shift slots whenever a move is
    disabled or out of PP, silently misaligning observations and actions.)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon, Status
from poke_env.data import GenData

TEAM_SIZE = 6  # Pokemon per side (random battles). Used for "remaining" ratios.
N_MOVES = 4

BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
MAX_BOOST = 6.0  # stat stages range from -6 to +6
STATUSES = (Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX)
MOVE_FEATURES = (
    "base_power",
    "effectiveness",
    "stab",
    "accuracy",
    "damage_score",
    "pp_fraction",
    "available",
)

# Normalisation constants (values above these are clipped to 1.0).
BASE_POWER_NORM = 150.0
EFFECTIVENESS_NORM = 4.0  # type multipliers are 0, 0.25, 0.5, 1, 2 or 4
DAMAGE_SCORE_NORM = 8.0  # e.g. 120 BP * 2x * 1.5 STAB = 3.6; 4x STAB hits ~5-9


def _pokemon_feature_names(prefix: str) -> List[str]:
    return (
        [f"{prefix}_hp"]
        + [f"{prefix}_boost_{stat}" for stat in BOOST_STATS]
        + [f"{prefix}_status_{status.name.lower()}" for status in STATUSES]
    )


def _build_feature_names() -> List[str]:
    names = _pokemon_feature_names("own") + _pokemon_feature_names("opp")
    names += ["opponent_threat", "speed_advantage", "can_tera"]
    for i in range(N_MOVES):
        names += [f"move{i}_{feature}" for feature in MOVE_FEATURES]
    for i in range(TEAM_SIZE):
        names += [f"team{i}_hp", f"team{i}_best_damage"]
    names += ["own_remaining", "opp_remaining"]
    return names


FEATURE_NAMES: List[str] = _build_feature_names()
N_FEATURES: int = len(FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _effectiveness(attack_type, defender: Optional[Pokemon], type_chart) -> float:
    """Type multiplier of `attack_type` against `defender` (1.0 if unknown)."""
    if defender is None:
        return 1.0
    return attack_type.damage_multiplier(
        defender.type_1, defender.type_2, type_chart=type_chart
    )


def _offence(move: Move, user: Pokemon, target: Optional[Pokemon], type_chart):
    """Return (effectiveness, stab, expected_damage) for a move.

    expected_damage = (base_power / 100) * effectiveness * STAB * accuracy.
    It is a cheap damage proxy, not a damage calculator: it ignores stats,
    items, abilities and weather. Status moves return zeros. Moves with a
    variable or fixed damage formula report base power 0 in the data, so they
    also look like zero-damage moves.
    """
    if move.category == MoveCategory.STATUS:
        return 0.0, 0.0, 0.0
    effectiveness = _effectiveness(move.type, target, type_chart)
    has_stab = move.type in user.types
    stab_multiplier = user.stab_multiplier if has_stab else 1.0
    damage = (move.base_power / 100.0) * effectiveness * stab_multiplier * move.accuracy
    return effectiveness, float(has_stab), damage


def _move_slots(battle: AbstractBattle) -> List[Move]:
    """The moves behind action indices 6-9, mirroring SinglesEnv.action_to_order."""
    mon = battle.active_pokemon
    if mon is None:
        return []
    known = list(mon.moves.values())[:N_MOVES]
    available = battle.available_moves
    # Forced single move that is not one of the known moves (Struggle, Recharge...)
    if len(available) == 1 and available[0].id not in {m.id for m in known}:
        return list(available)
    return known


def _pokemon_block(mon: Optional[Pokemon]) -> List[float]:
    """HP fraction, stat boosts and status flags of one Pokemon."""
    if mon is None:
        return [0.0] * (1 + len(BOOST_STATS) + len(STATUSES))
    boosts = [mon.boosts.get(stat, 0) / MAX_BOOST for stat in BOOST_STATS]
    status = [float(mon.status == s) for s in STATUSES]
    return [float(mon.current_hp_fraction)] + boosts + status


def _effective_speed(mon: Pokemon) -> float:
    """Base speed adjusted for stat stages and paralysis (a rough estimate)."""
    stage = mon.boosts.get("spe", 0)
    stage_multiplier = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
    if mon.status == Status.PAR:
        stage_multiplier *= 0.5
    return mon.base_stats.get("spe", 0) * stage_multiplier


def _speed_advantage(own: Optional[Pokemon], opp: Optional[Pokemon]) -> float:
    """own_speed / (own_speed + opp_speed): 0.5 = tie, near 1.0 = we outspeed."""
    if own is None or opp is None:
        return 0.5
    own_speed, opp_speed = _effective_speed(own), _effective_speed(opp)
    total = own_speed + opp_speed
    return own_speed / total if total > 0 else 0.5


def _opponent_threat(own: Optional[Pokemon], opp: Optional[Pokemon], type_chart) -> float:
    """Best STAB-type multiplier the opponent has against our active Pokemon."""
    if own is None or opp is None:
        return 0.0
    multipliers = [
        t.damage_multiplier(own.type_1, own.type_2, type_chart=type_chart)
        for t in opp.types
        if t is not None
    ]
    return min(max(multipliers, default=1.0) / EFFECTIVENESS_NORM, 1.0)


def _move_block(
    battle: AbstractBattle, own: Optional[Pokemon], opp: Optional[Pokemon], type_chart
) -> List[float]:
    """N_MOVES x len(MOVE_FEATURES) values, aligned with action indices 6-9."""
    available_ids = {m.id for m in battle.available_moves}
    slots = _move_slots(battle) if own is not None else []
    block: List[float] = []
    for i in range(N_MOVES):
        if i >= len(slots):
            block += [0.0] * len(MOVE_FEATURES)  # empty slot
            continue
        move = slots[i]
        effectiveness, stab, damage = _offence(move, own, opp, type_chart)
        pp_fraction = move.current_pp / move.max_pp if move.max_pp else 0.0
        block += [
            min(move.base_power / BASE_POWER_NORM, 1.0),
            min(effectiveness / EFFECTIVENESS_NORM, 1.0),
            stab,
            min(max(move.accuracy, 0.0), 1.0),
            min(damage / DAMAGE_SCORE_NORM, 1.0),
            min(max(pp_fraction, 0.0), 1.0),
            float(move.id in available_ids),
        ]
    return block


def _team_block(battle: AbstractBattle, opp: Optional[Pokemon], type_chart) -> List[float]:
    """HP and best expected damage per team slot, aligned with actions 0-5."""
    mons = list(battle.team.values())
    block: List[float] = []
    for i in range(TEAM_SIZE):
        if i >= len(mons) or mons[i].fainted:
            block += [0.0, 0.0]
            continue
        mon = mons[i]
        best = max(
            (_offence(m, mon, opp, type_chart)[2] for m in mon.moves.values()),
            default=0.0,
        )
        block += [float(mon.current_hp_fraction), min(best / DAMAGE_SCORE_NORM, 1.0)]
    return block


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def embed_battle(battle: AbstractBattle) -> np.ndarray:
    """Encode `battle` from the point of view of the player that owns it.

    `battle.team` is always the agent's own side and `battle.opponent_team` the
    other one, so the same function works for both agents in self-play.
    Returns a float32 array of shape (N_FEATURES,) with values in [-1, 1].
    """
    type_chart = GenData.from_gen(battle.gen).type_chart
    own, opp = battle.active_pokemon, battle.opponent_active_pokemon

    own_fainted = sum(m.fainted for m in battle.team.values())
    opp_fainted = sum(m.fainted for m in battle.opponent_team.values())

    parts: List[float] = []
    parts += _pokemon_block(own)
    parts += _pokemon_block(opp)
    parts += [
        _opponent_threat(own, opp, type_chart),
        _speed_advantage(own, opp),
        float(bool(battle.can_tera)),
    ]
    parts += _move_block(battle, own, opp, type_chart)
    parts += _team_block(battle, opp, type_chart)
    parts += [
        max(TEAM_SIZE - own_fainted, 0) / TEAM_SIZE,
        max(TEAM_SIZE - opp_fainted, 0) / TEAM_SIZE,
    ]

    vector = np.asarray(parts, dtype=np.float32)
    if vector.shape != (N_FEATURES,):  # layout and builder drifted apart
        raise RuntimeError(f"embedding has {vector.shape[0]} values, expected {N_FEATURES}")
    return np.clip(vector, -1.0, 1.0)
