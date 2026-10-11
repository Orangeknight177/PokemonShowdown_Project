# Gen 9 Random Battle bot

This project trains and runs a Pokémon Showdown agent with `poke-env` and
Stable-Baselines3. It supports parallel local battles, PPO/A2C training,
self-play, resumable checkpoints, baseline evaluation, and challenge play.
No account or battle dataset is needed for local training.

## Project layout

```text
pokebot/       battle features, rewards, environments, player and policy
train.py       PPO/A2C training and checkpointing
play.py        evaluation, challenge, accept, and ladder modes
smoke_test.py  local-server integration sanity check
```

## 1. Install prerequisites on Windows

Python 3.15 works with this project's dependencies on the current Windows
setup. Install the CPU-only Torch wheel from PyTorch's index first; the normal
PyPI index does not currently provide the needed Torch wheel for Python 3.15.
The commands below were checked with Torch 2.14.1+cpu, Stable-Baselines3 2.9.0,
poke-env 0.16.1, and NumPy 2.5.4. This setup uses CPU training. See the
[PyTorch install guide](https://pytorch.org/get-started/locally/) if you want
to configure a GPU build instead.

Create a virtual environment with the full path to your Python 3.15 executable
(replace the example path with yours):

```powershell
$pythonExe = "C:\Path\To\Python315\python.exe"
& $pythonExe -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Node.js and npm are also needed for the local Showdown server. Clone and set it
up in a separate folder:

```powershell
$showdownDir = Join-Path (Get-Location) "..\PokemonShowdown_Server"
git clone https://github.com/smogon/pokemon-showdown.git $showdownDir
Set-Location $showdownDir
npm install
Copy-Item config\config-example.js config\config.js
$configPath = ".\config\config.js"
$configText = Get-Content $configPath -Raw
$configText = $configText.Replace("exports.bindaddress = '0.0.0.0';", "exports.bindaddress = '127.0.0.1';")
Set-Content $configPath $configText
node pokemon-showdown start 8765 --no-security
```

Keep that server terminal running. The bot defaults to `127.0.0.1:8765`.
`--no-security` is for a private local training server only; do not expose it
to the internet.

## 2. Check the local connection

In a second PowerShell terminal, return to this project folder and run:

```powershell
.\.venv\Scripts\python.exe .\smoke_test.py --battles 3
```

This plays random legal actions against a built-in heuristic and checks that
the environment, observation mask, rewards, and battle player work together.

## 3. Train and keep improving

Start with four parallel battles against the built-in heuristic:

```powershell
.\.venv\Scripts\python.exe .\train.py --opponent heuristic --timesteps 300000 --num-envs 4 --save-dir runs\v1
```

Continue from the latest checkpoint later; `--timesteps` is additional
training time when resuming:

```powershell
.\.venv\Scripts\python.exe .\train.py --opponent heuristic --timesteps 300000 --num-envs 4 --save-dir runs\v1 --resume latest
```

Self-play makes both sides learn from the same policy. It needs SuperSuit,
which is included in `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe .\train.py --opponent self --timesteps 500000 --num-envs 4 --save-dir runs\v1-selfplay --resume runs\v1\final_model.zip
```

Checkpoints are written under `checkpoints\`; the final checkpoint is
`final_model.zip`. Training improves the policy at PPO/A2C update intervals,
not in the middle of an active turn. Keep evaluating against fixed opponents
so you can tell whether the latest model is actually stronger.

## 4. Evaluate and play

Evaluate against the built-in random, max-power, and heuristic opponents:

```powershell
.\.venv\Scripts\python.exe .\play.py eval --checkpoint runs\v1\final_model.zip --n-battles 100
```

For a local human challenge, start the local server and use a bot name:

```powershell
.\.venv\Scripts\python.exe .\play.py accept --checkpoint runs\v1\final_model.zip --username PokeRLBot --n-battles 3
```

The bot uses a random local name when `--username` is omitted. Official-server
play is opt-in and prompts for the password without echoing it; the password
is never accepted as a command-line argument:

```powershell
.\.venv\Scripts\python.exe .\play.py accept --checkpoint runs\v1\final_model.zip --showdown-official --username YourBotName --n-battles 3
```

Use a dedicated account if you later connect to the official server. Check
Pokémon Showdown's rules and ask staff before using automation on the public
ladder. The project defaults to one concurrent live battle; training uses
multiple local battles through `--num-envs`.

## Configuration

Useful training options include:

* `--opponent random|maxpower|heuristic|self`
* `--num-envs N` for the number of parallel local training battles
* `--checkpoint-freq N` for checkpoint frequency
* `--algo ppo|a2c`
* `--tensorboard-log PATH` for optional training curves (install TensorBoard
  separately)

The default format is `gen9randombattle`. The observation has 73 values and
an action mask for the 26 Gen 9 Singles actions. The reward combines HP change,
faints, status, and the final result. The model only sees battle information
available to its player; unseen opponent Pokémon and moves are hidden.

During play, an inference-time rule layer also filters moves blocked by a
revealed ability (or by a species' sole possible ability). It covers type
immunities such as Levitate, Sap Sipper, Storm Drain, and Volt Absorb, as well
as Soundproof, Bulletproof, Overcoat, Wind Rider, Wonder Guard, Good as Gold,
Magic Bounce, and abilities that block priority moves. This helps old
checkpoints immediately; it is a strategy rule, not automatic online training.
Abilities with multiple possible choices are not assumed until revealed.

## Limits

This is a learning baseline, not a strong competitive engine. Its compact
observation omits some useful context and the policy can need substantial
training. A single shared self-play policy can cycle, so compare checkpoints
against fixed baselines before promoting a newer model. Team preview and
formats other than six-Pokémon singles need additional handling.
