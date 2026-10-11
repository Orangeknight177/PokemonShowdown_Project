"""Inference-time safety rules for known ability immunities.

These rules sit on top of the trained policy. They do not alter its weights;
they prevent it from selecting moves that are certain to be blocked by the
opponent's revealed ability (or its only possible ability for that species).
"""

from __future__ import annotations

from poke_env.battle import AbstractBattle, Effect, Move, MoveCategory, Pokemon, Target
from poke_env.data import GenData

TYPE_IMMUNITIES = {
    "levitate": {"ground"}, "eartheater": {"ground"},
    "waterabsorb": {"water"}, "stormdrain": {"water"}, "dryskin": {"water"},
    "voltabsorb": {"electric"}, "lightningrod": {"electric"}, "motordrive": {"electric"},
    "sapsipper": {"grass"}, "flashfire": {"fire"}, "wellbakedbody": {"fire"},
}
MOVE_FLAG_IMMUNITIES = {
    "soundproof": "sound", "bulletproof": "bullet",
    "overcoat": "powder", "windrider": "wind",
}
ABILITY_IGNORING = {"moldbreaker", "teravolt", "turboblaze"}
MOVES_IGNORING_ABILITIES = {"moongeistbeam", "photongeyser", "sunsteelstrike"}
PRIORITY_BLOCKING_ABILITIES = {"dazzling", "queenlymajesty", "armortail"}


def _id(value: str | None) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _effective_known_ability(defender: Pokemon) -> str:
    """Use a revealed ability or the species' ability if it has only one."""
    ability = _id(defender.ability)
    if ability:
        return ability
    possible = {_id(value) for value in defender.possible_abilities} - {""}
    return next(iter(possible)) if len(possible) == 1 else ""


def _ability_is_suppressed(battle: AbstractBattle, defender: Pokemon) -> bool:
    if Effect.GASTRO_ACID in defender.effects:
        return True
    return any(
        pokemon is not None and _id(pokemon.ability) == "neutralizinggas"
        for pokemon in battle.all_active_pokemons
    )


def move_is_blocked_by_known_ability(
    move: Move, battle: AbstractBattle, attacker: Pokemon, defender: Pokemon
) -> bool:
    """Whether the opponent's active ability prevents this move working."""
    ability = _effective_known_ability(defender)
    if not ability or _ability_is_suppressed(battle, defender):
        return False

    attacker_ability = _id(attacker.ability)
    if (
        attacker_ability in ABILITY_IGNORING
        or move.id in MOVES_IGNORING_ABILITIES
        or (attacker_ability == "myceliummight" and move.category == MoveCategory.STATUS)
    ):
        return False

    move_type = move.type.name.lower() if move.type is not None else ""
    if move_type in TYPE_IMMUNITIES.get(ability, set()):
        return True

    flags = move.entry.get("flags", {})
    blocked_flag = MOVE_FLAG_IMMUNITIES.get(ability)
    if blocked_flag and flags.get(blocked_flag):
        return True

    if ability == "wonderguard" and move.category != MoveCategory.STATUS:
        chart = GenData.from_gen(battle.gen).type_chart
        if move.type is not None and move.type.damage_multiplier(
            defender.type_1, defender.type_2, type_chart=chart
        ) <= 1:
            return True

    targets_opponent = move.target in {
        Target.ADJACENT_FOE, Target.ALL_ADJACENT_FOES, Target.ANY,
        Target.NORMAL, Target.RANDOM_NORMAL,
    }
    if (
        ability in {"goodasgold", "magicbounce"}
        and move.category == MoveCategory.STATUS
        and targets_opponent
    ):
        return True
    return ability in PRIORITY_BLOCKING_ABILITIES and move.priority > 0


def mask_ability_blocked_moves(battle: AbstractBattle, mask):
    """Copy an action mask and disable move choices blocked by known abilities."""
    adjusted = mask.copy()
    attacker = battle.active_pokemon
    defender = battle.opponent_active_pokemon
    if attacker is None or defender is None:
        return adjusted

    moves = list(attacker.moves.values())[:4]
    available = battle.available_moves
    if len(available) == 1 and available[0].id not in {move.id for move in moves}:
        moves = list(available)

    for slot, move in enumerate(moves[:4]):
        action = 6 + slot
        if action < len(adjusted) and adjusted[action] and move_is_blocked_by_known_ability(
            move, battle, attacker, defender
        ):
            adjusted[action] = 0

    # A forced-move edge case should never turn the action mask fully empty.
    return adjusted if adjusted.any() else mask
