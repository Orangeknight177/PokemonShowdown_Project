#!/usr/bin/env python3
"""Load a trained checkpoint and play matches on a Showdown server.

Modes
    eval       play N battles against each scripted baseline and print win rates
    accept     wait for challenges (from you in the browser, or another bot)
    challenge  challenge a specific user to N battles
    ladder     queue up on the ladder for N games

Examples (local server on localhost:8000)
    python play.py eval --checkpoint runs/v1/final_model.zip --n-battles 100
    python play.py accept --checkpoint runs/v1/final_model.zip --username PokeRLBot --n-battles 3
    python play.py challenge --checkpoint runs/v1/final_model.zip --opponent SomeUser --n-battles 1

For `accept`, open http://localhost:8000, log in under any other name and
challenge the bot's username (gen9randombattle by default).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ShowdownServerConfiguration

from pokebot.config import DEFAULT_BATTLE_FORMAT, DEFAULT_HOST, DEFAULT_PORT, make_server_configuration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["eval", "accept", "challenge", "ladder"])
    p.add_argument("--checkpoint", required=True, help="path to a checkpoint .zip written by train.py")
    p.add_argument("--algo", default="ppo", choices=["ppo", "a2c"], help="algorithm the checkpoint was trained with")
    p.add_argument("--battle-format", default=DEFAULT_BATTLE_FORMAT)
    p.add_argument("--n-battles", type=int, default=10)
    p.add_argument("--opponent", default=None, help="username to challenge (mode: challenge)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions from the policy instead of taking the most likely one")
    p.add_argument("--debug", action="store_true",
                   help="show Showdown connection and battle messages during evaluation")
    p.add_argument("--device", default="cpu")

    g = p.add_argument_group("server")
    g.add_argument("--host", default=DEFAULT_HOST)
    g.add_argument("--port", type=int, default=DEFAULT_PORT)
    g.add_argument("--username", default=None, help="account name (random if omitted; needed for 'accept')")
    g.add_argument(
        "--password-env",
        default=None,
        metavar="NAME",
        help="read the official-server password from this environment variable; otherwise prompt securely",
    )
    g.add_argument("--showdown-official", action="store_true",
                   help="connect to the official Showdown server (requires --username)")
    return p.parse_args()


async def run_eval(agent, args, server) -> None:
    baselines = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    opponents = [
        cls(battle_format=args.battle_format, server_configuration=server,
            max_concurrent_battles=10)
        for cls in baselines
    ]
    print(
        f"Starting evaluation: {args.n_battles} battle(s) against each of "
        f"{len(opponents)} opponents.",
        flush=True,
    )
    for opp in opponents:
        print(f"Playing {type(opp).__name__}...", flush=True)
        await agent.battle_against(opp, n_battles=args.n_battles)
        # Counted from the opponent's side: its losses are our wins.
        wins = opp.n_lost_battles
        total = opp.n_finished_battles
        print(
            f"{type(opp).__name__:>22}: "
            f"{100 * wins / max(total, 1):5.1f}% ({wins}/{total})",
            flush=True,
        )


async def run(args) -> None:
    from pokebot.policy import load_policy_player  # torch import happens here

    if args.showdown_official:
        if not args.username:
            raise SystemExit("--showdown-official needs --username")
        password = os.environ.get(args.password_env) if args.password_env else None
        if not password:
            password = getpass.getpass(f"Password for {args.username}: ")
        if not password:
            raise SystemExit("A password is required for the official server")
        server = ShowdownServerConfiguration
        account = AccountConfiguration(args.username, password)
    else:
        server = make_server_configuration(args.host, args.port)
        account = (
            AccountConfiguration(args.username, None)
            if args.username
            else AccountConfiguration.generate("PokeRLBot", rand=True)
        )
    agent = load_policy_player(
        args.checkpoint,
        algo=args.algo,
        deterministic=not args.stochastic,
        device=args.device,
        account_configuration=account,
        battle_format=args.battle_format,
        server_configuration=server,
        max_concurrent_battles=10 if args.mode == "eval" else 1,
    )
    print(f"playing as {agent.username!r} on {server.websocket_url}")

    if args.mode == "eval":
        await run_eval(agent, args, server)
    elif args.mode == "accept":
        print(f"waiting for {args.n_battles} challenge(s)...")
        await agent.accept_challenges(None, args.n_battles)
    elif args.mode == "challenge":
        if not args.opponent:
            raise SystemExit("--opponent is required for 'challenge'")
        await agent.send_challenges(args.opponent, args.n_battles)
    else:  # ladder
        await agent.ladder(args.n_battles)

    if args.mode != "eval":
        print(f"finished {agent.n_finished_battles} battles, won {agent.n_won_battles}")


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
