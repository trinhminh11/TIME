# TIME: Trajectory-level Information Maximization and Exploration

TIME is an unsupervised reinforcement learning algorithm that maximizes information at the trajectory level for enhanced exploration. The algorithm is trained using `agent=ours` and demonstrates superior exploration efficiency compared to existing methods.

This repository contains the implementation of TIME and is adapted from the [URL Benchmark](https://github.com/rll-research/url_benchmark) codebase. The code also includes implementations of other baseline unsupervised RL algorithms for comparison purposes.

**This repository will be used for paper publication.**

## Requirements
We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to use `uv`:
```sh
uv sync
```
After the installation ends you can activate your environment with
```sh
source .venv/bin/activate
```

## Implemented Agents
| Agent | Command | Paper |
|---|---|---|
| TIME (Ours) | `agent=ours` |  |
| ICM | `agent=icm` | [paper](https://arxiv.org/abs/1705.05363)|
| ProtoRL | `agent=proto` | [paper](https://arxiv.org/abs/2102.11271)|
| DIAYN | `agent=diayn` | [paper](https://arxiv.org/abs/1802.06070)|
| APT(ICM) | `agent=icm_apt` | [paper](https://arxiv.org/abs/2103.04551)|
| APT(Ind) | `agent=ind_apt` | [paper](https://arxiv.org/abs/2103.04551)|
| APS | `agent=aps` | [paper](http://proceedings.mlr.press/v139/liu21b.html)|
| SMM | `agent=smm` | [paper](https://arxiv.org/abs/1906.05274) |
| RND | `agent=rnd` | [paper](https://arxiv.org/abs/1810.12894) |
| Disagreement | `agent=disagreement` | [paper](https://arxiv.org/abs/1906.04161) |

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `hopper` | `hop`, `stand` |
| `cheetah` | `run` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `humanoid` | `stand`, `walk`, `run` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |
| `ant` | `temp` |
| `antmaze` | `umaze`, `medium_play`, `medium_diverse`, `large_play`, `large_diverse` |


## Domain observation mode
Each domain supports two observation modes: states and pixels.
| Model | Command |
|---|---|
| states | `obs_type=states` |
| pixels | `obs_type=pixels` |


## Instructions
### Pre-training
To run pre-training use the `pretrain.py` script. To train TIME, run:
```sh
python pretrain.py agent=ours domain=walker
```
or use any of the baseline algorithms, for example:
```sh
python pretrain.py agent=icm domain=walker
```
This script will produce several agent snapshots after training for `100k`, `500k`, `1M`, and `2M` frames. The snapshots will be stored under the following directory:
```sh
./pretrained_models/<obs_type>/<domain>/<agent>/
```
For example:
```sh
./pretrained_models/states/walker/icm/
```

For AntMaze pre-training, use:
```sh
python pretrain.py agent=diayn domain=antmaze obs_type=states
```
AntMaze currently supports `obs_type=states` only.

### Fine-tuning
Once you have pre-trained your method, you can use the saved snapshots to initialize the `DDPG` agent and fine-tune it on a downstream task. For example, to fine-tune the pre-trained TIME model on `walker_run`, run:
```sh
python finetune.py pretrained_agent=ours task=walker_run snapshot_ts=1000000 obs_type=states
```
This will load a snapshot stored in `./pretrained_models/states/walker/ours/snapshot_1000000.pt`, initialize `DDPG` with it (both the actor and critic), and start training on `walker_run` using the extrinsic reward of the task.

You can also fine-tune other baseline methods:
```sh
python finetune.py pretrained_agent=icm task=walker_run snapshot_ts=1000000 obs_type=states
```

For methods that use skills, include the agent, and the `reward_free` tag to false.
```sh
python finetune.py pretrained_agent=smm task=walker_run snapshot_ts=1000000 obs_type=states agent=smm reward_free=false
```

### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in a form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```
