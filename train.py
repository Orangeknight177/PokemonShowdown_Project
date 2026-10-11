#!/usr/bin/env python3
"""Train (or keep training) the Pokemon Showdown agent with PPO / A2C.

Examples
    # Train against the built-in heuristic bot, saving a checkpoint every 25k steps
    python train.py --opponent heuristic --timesteps 300000 --num-envs 4 --save-dir runs/v1

    # Continue the same run later from its newest checkpoint (adds 300k more steps)
    python train.py --opponent heuristic --timesteps 300000 --save-dir runs/v1 --resume latest

    # Switch to self-play (both sides share the learning policy), warm-starting
    # from the checkpoint above
    python train.py --opponent self --timesteps 500000 --save-dir runs/v1_selfplay \\
        --resume runs/v1/checkpoints/ppo_300000_steps.zip

Start a local server first (see README.md):  node pokemon-showdown start --no-security
"""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path
from typing import Optional

from pokebot.config import DEFAULT_BATTLE_FORMAT, DEFAULT_HOST, DEFAULT_PORT, make_server_configuration
from pokebot.env import BASELINE_OPPONENTS, make_baseline_env, make_selfplay_vec_env
from pokebot.rewards import RewardConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("environment")
    g.add_argument("--opponent", default="heuristic", choices=sorted(BASELINE_OPPONENTS) + ["self"],
                   help="scripted baseline, or 'self' for self-play (default: heuristic)")
    g.add_argument("--battle-format", default=DEFAULT_BATTLE_FORMAT)
    g.add_argument("--host", default=DEFAULT_HOST, help="Showdown server host")
    g.add_argument("--port", type=int, default=DEFAULT_PORT, help="Showdown server port")
    g.add_argument("--num-envs", type=int, default=4, help="parallel battles (each uses 2 server connections)")

    g = p.add_argument_group("training")
    g.add_argument("--algo", default="ppo", choices=["ppo", "a2c"])
    g.add_argument("--timesteps", type=int, default=200_000,
                   help="decisions to train for (ADDITIONAL steps when resuming)")
    g.add_argument("--checkpoint-freq", type=int, default=25_000, help="save a checkpoint every N timesteps")
    g.add_argument("--save-dir", default="runs/default", help="checkpoints/, final_model.zip, config.json go here")
    g.add_argument("--resume", default=None,
                   help="checkpoint .zip to continue from, or 'latest' for the newest in <save-dir>/checkpoints")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default="cpu", help="cpu is fastest for this small MLP; use cuda if you scale it up")
    g.add_argument("--tensorboard-log", default=None, help="optional TensorBoard log directory")

    g = p.add_argument_group("hyperparameters (ignored for loaded checkpoints, which keep their own)")
    g.add_argument("--learning-rate", type=float, default=3e-4)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--ent-coef", type=float, default=0.01)
    g.add_argument("--rollout-size", type=int, default=3072, help="decisions collected per update across all envs")
    g.add_argument("--batch-size", type=int, default=128, help="PPO minibatch size")
    g.add_argument("--net-arch", default="128,128", help="hidden layer sizes, e.g. 128,128")

    g = p.add_argument_group("reward weights (see pokebot/rewards.py)")
    g.add_argument("--hp-weight", type=float, default=1.0)
    g.add_argument("--faint-weight", type=float, default=2.0)
    g.add_argument("--status-weight", type=float, default=0.5)
    g.add_argument("--win-bonus", type=float, default=30.0)
    return p.parse_args()


def reward_config_from_args(args: argparse.Namespace) -> RewardConfig:
    """Symmetric weights: what we gain for hurting them we lose when they hurt us."""
    return RewardConfig(
        damage_dealt=args.hp_weight, damage_taken=args.hp_weight,
        faint_dealt=args.faint_weight, faint_taken=args.faint_weight,
        status_dealt=args.status_weight, status_taken=args.status_weight,
        win_bonus=args.win_bonus, loss_penalty=args.win_bonus,
    )


def build_vec_env(args: argparse.Namespace, reward_config: RewardConfig):
    server = make_server_configuration(args.host, args.port)
    if args.opponent == "self":
        return make_selfplay_vec_env(
            args.num_envs, battle_format=args.battle_format,
            server_configuration=server, reward_config=reward_config,
        )
    # SubprocVecEnv runs each battle env in its own process. The factory must be
    # picklable, hence functools.partial over a module-level function.
    from stable_baselines3.common.vec_env import SubprocVecEnv

    factory = partial(
        make_baseline_env, args.opponent, battle_format=args.battle_format,
        server_configuration=server, reward_config=reward_config,
    )
    return SubprocVecEnv([factory for _ in range(args.num_envs)])


def find_latest_checkpoint(directory: Path, prefix: str) -> Optional[Path]:
    """Newest `<prefix>_<steps>_steps.zip` in `directory` (the CheckpointCallback naming)."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    found = [(int(m.group(1)), f) for f in directory.glob("*.zip") if (m := pattern.match(f.name))]
    return max(found)[1] if found else None


def main() -> None:
    args = parse_args()

    # Heavy imports after argument parsing so `--help` is instant.
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecMonitor

    from pokebot.policy import ALGOS, MaskedActorCriticPolicy, load_model

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    reward_config = reward_config_from_args(args)
    vec_env = build_vec_env(args, reward_config)
    if args.opponent == "self":
        vec_env = VecMonitor(vec_env)  # episode stats (the baseline path uses Monitor per env)

    # ---- model: resume from a checkpoint or start fresh ---------------------
    resume_path: Optional[Path] = None
    if args.resume == "latest":
        resume_path = find_latest_checkpoint(ckpt_dir, args.algo)
        if resume_path is None:
            print(f"no checkpoint found in {ckpt_dir}; starting a new run")
    elif args.resume:
        resume_path = Path(args.resume)

    if resume_path is not None:
        print(f"resuming from {resume_path}")
        model = load_model(str(resume_path), algo=args.algo, env=vec_env, device=args.device)
        if args.tensorboard_log:
            model.tensorboard_log = args.tensorboard_log
    else:
        n_steps = max(args.rollout_size // vec_env.num_envs, 1)  # per sub-environment
        kwargs = dict(
            learning_rate=args.learning_rate, n_steps=n_steps, gamma=args.gamma,
            ent_coef=args.ent_coef, device=args.device, seed=args.seed,
            tensorboard_log=args.tensorboard_log, verbose=1,
            policy_kwargs=dict(net_arch=[int(x) for x in args.net_arch.split(",")]),
        )
        if args.algo == "ppo":
            kwargs["batch_size"] = min(args.batch_size, n_steps * vec_env.num_envs)
        model = ALGOS[args.algo](MaskedActorCriticPolicy, vec_env, **kwargs)

    # ---- checkpoints --------------------------------------------------------
    # The callback fires once per vec-env step (= num_envs timesteps), so divide.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // vec_env.num_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=args.algo,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=resume_path is None,  # keep counting when resuming
            tb_log_name=f"{args.algo}_{args.opponent}",
        )
    finally:
        # Also runs on Ctrl+C / crashes, so progress is never lost.
        final_path = save_dir / "final_model"
        model.save(str(final_path))
        print(f"saved {final_path}.zip after {model.num_timesteps} total timesteps")
        vec_env.close()


if __name__ == "__main__":  # required: SubprocVecEnv workers re-import this module
    main()
