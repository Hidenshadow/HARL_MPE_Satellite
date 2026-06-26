# Satellite-MPE MAE Benchmark

This repository provides a compact, paper-oriented codebase for the Satellite-MPE cooperative multi-agent benchmark. It keeps the baseline on-policy algorithms `happo`, `hatrpo`, and `mappo`, together with the MAE variants `MAE_happo`, `MAE_hatrpo`, and `MAE_mappo`.

The benchmark environment is implemented in:

```text
examples/PettingZoo/pettingzoo/mpe/Satellite/simple_satellite.py
```

The project is derived from HARL and PettingZoo MPE, but has been trimmed for a focused open-source release: experiment outputs, plotting scripts, external simulator artifacts, and unused algorithm variants are not part of this release.

## Highlights

- Satellite-MPE benchmark with symmetric and asymmetric agent configurations.
- Twelve built-in map configurations covering small, medium, and large settings.
- PettingZoo parallel API support for direct environment use.
- One-command training through `examples/train.py`.
- Retained algorithms: `happo`, `hatrpo`, `mappo`, `MAE_happo`, `MAE_hatrpo`, `MAE_mappo`.
- Reproducible run folders containing logs, saved configs, and model checkpoints.

## Repository Structure

```text
HARL/
  examples/
    train.py                                      # Training entry point
    PettingZoo/pettingzoo/mpe/Satellite/
      simple_satellite.py                        # Satellite-MPE benchmark
  harl/
    algorithms/actors/                           # Baseline and MAE actor updates
    algorithms/critics/                          # Centralized critics
    common/buffers/                              # On-policy rollout buffers
    configs/algos_cfgs/                          # Algorithm YAML configs
    configs/envs_cfgs/pettingzoo_mpe.yaml        # Satellite-MPE env config
    envs/pettingzoo_mpe/                         # HARL wrapper for PettingZoo MPE
    runners/                                     # Training loops
```

## Installation

Create an environment and install the repository in editable mode:

```bash
conda create -n satellite_mpe python=3.10 -y
conda activate satellite_mpe

# Install PyTorch following https://pytorch.org/get-started/locally/
# Example CPU install:
pip install torch

pip install -r requirements.txt
pip install -e HARL
```

The training script automatically prioritizes the local PettingZoo copy under `examples/PettingZoo`, so no separate PettingZoo checkout is required.

All commands below are written for the project root that contains the `HARL/` directory.

## Quick Start

List all built-in satellite maps:

```bash
python3 HARL/examples/train.py --list_satellite_maps
```

Run a short smoke test:

```bash
python3 HARL/examples/train.py \
  --algo MAE_mappo \
  --env pettingzoo_mpe \
  --map_name 2x2-4A \
  --exp_name smoke \
  --num_env_steps 4 \
  --episode_length 2 \
  --n_rollout_threads 1 \
  --n_eval_rollout_threads 1 \
  --eval_episodes 1 \
  --use_eval False \
  --cuda False \
  --log_dir /tmp/harl_satellite_smoke
```

Train one full Satellite-MPE run:

```bash
python3 HARL/examples/train.py \
  --algo MAE_mappo \
  --env pettingzoo_mpe \
  --map_name 2x2-4A \
  --exp_name satellite_2x2_4A
```

Run all maps with one algorithm:

```bash
for map in 2x2-4A 2x2-2A2C 3x2-6A 3x2-2A2B2C 4x2-8A 4x2-2A4B2C 3x3-9A 3x3-4A1B4C 4x3-12A 4x3-6A2B4C 4x4-16A 4x4-8A4B4C
do
  python3 HARL/examples/train.py \
    --algo MAE_mappo \
    --env pettingzoo_mpe \
    --map_name "$map" \
    --exp_name "satellite_$map"
done
```

Run all retained algorithms on one map:

```bash
for algo in happo hatrpo mappo MAE_happo MAE_hatrpo MAE_mappo
do
  python3 HARL/examples/train.py \
    --algo "$algo" \
    --env pettingzoo_mpe \
    --map_name 2x2-4A \
    --exp_name "2x2_4A_$algo"
done
```

## Satellite-MPE Benchmark

Satellite-MPE is a cooperative multi-agent target observation task. Agents move in a bounded 2D world, rotate their sensing direction, and observe moving targets within a limited field of view and sensing radius. A target is counted as covered only when at least two distinct agents observe it for consecutive steps. Covered targets are removed and contribute to the shared team reward.

### Action Space

The default action space is discrete with seven actions:

| Action | Meaning |
| ---: | :-- |
| 0 | Stay |
| 1 | Move left |
| 2 | Move right |
| 3 | Move down |
| 4 | Move up |
| 5 | Rotate clockwise by 15 deg |
| 6 | Rotate counter-clockwise by 15 deg |

### Observation

Each agent observes a fixed-size vector:

```text
[self velocity, self position, assigned region bounds,
 visible agents, visible targets]
```

The observation is padded to support a fixed neural network input size. For a map with `N` agents and `T` targets, the local observation dimension is:

```text
8 + 4 * (N - 1) + 3 * T
```

The centralized state used by the runner is the concatenation of all local observations.

### Reward

For each target:

1. Each agent checks whether the target is inside its sensing radius and field of view.
2. A target visit is accumulated only when at least two distinct agents observe it at the same step.
3. Once the accumulated visit count reaches 5, the target is removed and the team receives reward `+1`.

The HARL wrapper broadcasts the shared team reward to all agents.

## BSK Experiments

Coming soon.

## Map Configurations

`--map_name` selects the number of agents, targets, grid partition, agent sensing parameters, environment `max_cycles`, and training `episode_length`. If `--episode_length` is not manually overridden, it is synchronized with the selected map.

| Size | Map | Targets | Grid | Agent sensing setup | Episode length |
| :-- | :-- | --: | :-- | :-- | --: |
| Small | `2x2-4A` | 40 | `2x2` | All: 15 deg, radius 1.25 | 100 |
| Small | `2x2-2A2C` | 40 | `2x2` | Agents 0,3: 15 deg, 1.25; agents 1,2: 10 deg, 1.5 | 100 |
| Small | `3x2-6A` | 60 | `2x3` | All: 15 deg, radius 1.25 | 125 |
| Small | `3x2-2A2B2C` | 60 | `2x3` | Agents 0,5: 15 deg, 1.25; agents 1,4: 10 deg, 1.5; agents 2,3: 30 deg, 1.0 | 125 |
| Medium | `4x2-8A` | 80 | `2x4` | All: 15 deg, radius 1.25 | 150 |
| Medium | `4x2-2A4B2C` | 80 | `2x4` | Agents 0,7: 15 deg, 1.25; agents 1,6: 10 deg, 1.5; agents 2-5: 30 deg, 1.0 | 150 |
| Medium | `3x3-9A` | 90 | `3x3` | All: 15 deg, radius 1.25 | 150 |
| Medium | `3x3-4A1B4C` | 90 | `3x3` | Agents 1,3,5,7: 15 deg, 1.25; agents 0,2,6,8: 10 deg, 1.5; agent 4: 30 deg, 1.0 | 150 |
| Large | `4x3-12A` | 96 | `3x4` | All: 15 deg, radius 1.25 | 150 |
| Large | `4x3-6A2B4C` | 96 | `3x4` | Agents 1,3,5,6,8,10: 15 deg, 1.25; agents 0,2,9,11: 10 deg, 1.5; agents 4,7: 30 deg, 1.0 | 150 |
| Large | `4x4-16A` | 112 | `4x4` | All: 15 deg, radius 1.25 | 150 |
| Large | `4x4-8A4B4C` | 112 | `4x4` | Agents 1,2,4,7,8,11,13,14: 15 deg, 1.25; agents 0,3,12,15: 10 deg, 1.5; agents 5,6,9,10: 30 deg, 1.0 | 150 |

## Direct Environment Use

To use the environment without the training runner:

```bash
export PYTHONPATH=$PWD/HARL:$PWD/HARL/examples/PettingZoo:$PYTHONPATH
python3 - <<'PY'
from pettingzoo.mpe.simple_satellite import available_maps, parallel_env

print(available_maps())

env = parallel_env(map_name="2x2-4A")
obs = env.reset(seed=0)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)

print(env.agents)
print({agent: obs[agent].shape for agent in env.agents})
print(rewards)
env.close()
PY
```

## Configuration

Default algorithm hyperparameters are stored in:

```text
harl/configs/algos_cfgs/
```

Default Satellite-MPE environment settings are stored in:

```text
harl/configs/envs_cfgs/pettingzoo_mpe.yaml
```

Any YAML field can be overridden from the command line. For example:

```bash
python3 HARL/examples/train.py \
  --algo MAE_happo \
  --env pettingzoo_mpe \
  --map_name 3x3-4A1B4C \
  --seed 1 \
  --num_env_steps 1000000 \
  --n_rollout_threads 8 \
  --cuda True \
  --exp_name mae_happo_3x3_seed1
```

## Outputs

Training outputs are written under the configured `log_dir`, which defaults to:

```text
results/
```

Each run stores:

- `config.json`: resolved training, algorithm, and environment settings.
- `logs/`: TensorBoard event files.
- `models/`: saved policy and critic checkpoints.

Example TensorBoard command:

```bash
tensorboard --logdir results
```

## Algorithms

| Algorithm | Type | Parameter sharing default | Notes |
| :-- | :-- | :-- | :-- |
| `happo` | Baseline | No | Heterogeneous-agent PPO baseline |
| `hatrpo` | Baseline | No | Heterogeneous-agent TRPO baseline |
| `mappo` | Baseline | Yes | Shared-policy PPO baseline |
| `MAE_happo` | MAE variant | No | MAE extension of HAPPO |
| `MAE_hatrpo` | MAE variant | No | MAE extension of HATRPO |
| `MAE_mappo` | MAE variant | Yes | MAE extension of MAPPO |

## Reproducibility Checklist

For paper experiments, record the following for every run:

- Git commit or release tag.
- Algorithm name and YAML config.
- `map_name`.
- Random seed.
- `num_env_steps`, `episode_length`, and `n_rollout_threads`.
- Hardware and PyTorch/CUDA versions.
- Full generated `config.json` from the run folder.

The following command prints the final map registry used by the code:

```bash
python3 HARL/examples/train.py --list_satellite_maps
```

## Troubleshooting

If `harl` cannot be imported, run commands from the project root that contains `HARL/` or reinstall the package:

```bash
pip install -e HARL
```

If `pettingzoo.mpe.simple_satellite` cannot be imported in a standalone script, add the local PettingZoo package to `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/HARL:$PWD/HARL/examples/PettingZoo:$PYTHONPATH
```

If training starts but CUDA is not available, disable CUDA explicitly:

```bash
python3 HARL/examples/train.py --algo MAE_mappo --env pettingzoo_mpe --map_name 2x2-4A --cuda False --exp_name cpu_run
```

## Acknowledgements

This repository builds on the HARL codebase and the PettingZoo MPE environment interface. Please cite the original HARL and PettingZoo projects where appropriate.

## Citation

If you use this benchmark or codebase in a paper, please cite the accompanying paper. The camera-ready BibTeX entry should be added here once the paper metadata is public.

## License

This project follows the MIT license metadata inherited from the HARL package.
