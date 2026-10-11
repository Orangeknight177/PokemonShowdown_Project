"""Shared defaults and small helpers for talking to a Showdown server."""

from __future__ import annotations

from poke_env.ps_client import ServerConfiguration

# Random battles hand both players a random team, so no team building or
# team preview logic is needed. Change this (and supply a team) for other formats.
DEFAULT_BATTLE_FORMAT = "gen9randombattle"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# A local server started with `--no-security` never calls this URL, but
# poke-env's ServerConfiguration requires one.
_AUTH_URL = "https://play.pokemonshowdown.com/action.php?"


def make_server_configuration(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ServerConfiguration:
    """Build a poke-env ServerConfiguration for a (local) Showdown server."""
    return ServerConfiguration(f"ws://{host}:{port}/showdown/websocket", _AUTH_URL)
