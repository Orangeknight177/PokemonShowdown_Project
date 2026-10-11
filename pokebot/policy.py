"""stable-baselines3 pieces: action-masked policy and checkpoint loading.

Why a custom policy?
    Pokemon has many illegal actions at any time (fainted switch targets,
    moves out of PP, being trapped...). SinglesEnv exposes a 0/1 "action_mask"
    next to every observation. We feed only the "observation" part to the
    network and add -inf to the logits of illegal actions, so the agent can
    never choose (or waste probability mass on) an illegal action.

IMPORTANT: keep this class importable under the same module path
(`pokebot.policy.MaskedActorCriticPolicy`) for both training and playing.
SB3 stores the policy class by reference inside checkpoints, which is why the
training and play scripts live next to the `pokebot` package instead of
defining the policy inside `__main__`.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from pokebot.players import ActFn, PolicyPlayer

# DQN is intentionally not offered: its Q-network policy has no equivalent of
# the logit-masking hook below, and PPO / A2C are the better fit for this
# large, partially observable, mostly on-policy task.
ALGOS = {"ppo": PPO, "a2c": A2C}


class FeaturesExtractor(BaseFeaturesExtractor):
    """Picks the "observation" vector out of the {observation, action_mask} dict.

    The mask must NOT be fed to the network as an ordinary feature: it is
    applied to the action logits instead (see MaskedActorCriticPolicy).
    """

    def __init__(self, observation_space):
        super().__init__(
            observation_space, features_dim=observation_space["observation"].shape[0]
        )

    def forward(self, obs):
        return obs["observation"]


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy that forbids illegal actions via logit masking.

    SB3 reaches the action distribution through three different entry points
    (forward for rollouts, evaluate_actions for the loss, get_distribution for
    predict()). Each one stashes the mask from the observation dict first;
    `_get_action_dist_from_latent` then applies it.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("net_arch", [128, 128])
        kwargs.setdefault("features_extractor_class", FeaturesExtractor)
        super().__init__(*args, **kwargs)
        self._mask: Optional[torch.Tensor] = None

    def forward(self, obs, deterministic: bool = False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs):
        self._mask = obs["action_mask"]
        return super().get_distribution(obs)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        # legal action -> +0, illegal action -> -inf (probability exactly 0)
        penalty = torch.where(self._mask == 1, 0.0, float("-inf")).to(logits.dtype)
        return self.action_dist.proba_distribution(action_logits=logits + penalty)


def make_act_fn(policy: ActorCriticPolicy, deterministic: bool = True) -> ActFn:
    """Copy the small MLP to Python numbers and run inference without Torch.

    poke-env calls ``choose_move`` from its own background event-loop thread.
    Avoiding Torch in that callback works around native Torch crashes seen with
    the current Windows/Python runtime while preserving the trained network.
    """
    linear_layers = []
    for layer in policy.mlp_extractor.policy_net:
        if isinstance(layer, torch.nn.Linear):
            linear_layers.append((
                layer.weight.detach().cpu().tolist(),
                layer.bias.detach().cpu().tolist(),
            ))
        elif isinstance(layer, torch.nn.Tanh):
            linear_layers.append(None)
        else:
            raise TypeError(f"Unsupported policy layer for pure-Python inference: {type(layer).__name__}")

    action_layer = policy.action_net
    if not isinstance(action_layer, torch.nn.Linear):
        raise TypeError("Expected a linear discrete-action output layer")
    action_weights = action_layer.weight.detach().cpu().tolist()
    action_bias = action_layer.bias.detach().cpu().tolist()
    rng = random.Random()

    def linear(values, weights, biases):
        return [
            sum(weight * value for weight, value in zip(row, values)) + bias
            for row, bias in zip(weights, biases)
        ]

    def act(obs: np.ndarray, mask: np.ndarray) -> int:
        values = [float(value) for value in obs]
        for layer in linear_layers:
            if layer is None:
                values = [math.tanh(value) for value in values]
            else:
                values = linear(values, *layer)
        logits = linear(values, action_weights, action_bias)
        legal = [index for index, allowed in enumerate(mask) if allowed]
        if not legal:
            raise ValueError("The battle observation has no legal actions")
        if deterministic:
            return max(legal, key=lambda index: logits[index])

        top = max(logits[index] for index in legal)
        weights = [math.exp(logits[index] - top) for index in legal]
        return rng.choices(legal, weights=weights, k=1)[0]

    return act


def load_model(path: str, algo: str = "ppo", env=None, device: str = "cpu"):
    """Load a full SB3 checkpoint (.zip). Pass `env` to continue training."""
    return ALGOS[algo].load(path, env=env, device=device)


def load_policy_player(
    checkpoint: str,
    algo: str = "ppo",
    *,
    deterministic: bool = True,
    device: str = "cpu",
    **player_kwargs,
) -> PolicyPlayer:
    """Build a PolicyPlayer from a checkpoint; player_kwargs go to poke-env's Player."""
    model = load_model(checkpoint, algo=algo, device=device)
    return PolicyPlayer(make_act_fn(model.policy, deterministic), **player_kwargs)
