# Hector IsaacLab simulation 

## Before begin

### Install IsaacSim 4.5
See official [docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim) \
Use conda and pip install IsaacSim4.5. 

### Clone this repository
```bash
git clone git@github.com:jnskkmhr/IsaacLab.git -b devel 
```

### Pull usd asset via git LFS
```bash
cd source/isaaclab_assets/data/Robot/Hector
git lfs pull
```

### Install IsaacLab
```bash
conda activate env_isaaclab
cd ~/IsaacLab
./isaaclab.sh -i
```

### Build hector controller
```bash
git clone https://github.gatech.edu/GeorgiaTechLIDARGroup/HECTOR_HW_new.git -b devel/slope_terrain
./isaaclab.sh -p -m pip install -e {/path/to/hector/controller}
```

### Install RL libraries
We use modified library of rsl-rl and rl_games (recommended). 
```bash
git clone git@github.com:jnskkmhr/rl_games.git
git clone -b devel git@github.com:jnskkmhr/rsl_rl.git

# and install
./isaaclab.sh -p -m pip install -e {/path/to/rsl_rl}
./isaaclab.sh -p -m pip install -e {/path/to/rl_games}
./isaaclab.sh -p -m pip install ray
```

## Run code

### Run MPC

Only MPC

```bash
./isaaclab.sh -p scripts/biped/run_direct_env.py --task SteppingStone --num_envs 1 --max_trials 10 --episode_length 20
```

### Train RL 

#### (RSl-RL)
```bash
./isaaclab.sh -p scripts/biped/rsl_rl/train.py --task SteppingStone --num_envs 32 --video --headless
```
After a while, you will see the logs under `logs` directory (for example `logs/rsl_rl/ppo_rsl_rl_lstm_friction/2025-03-17_15-31-27`). 

#### (RL-Games)
```bash
./isaaclab.sh -p scripts/biped/rl_games/train.py --task SteppingStone --num_envs 32 --video --headless
```

### Inference
```bash
./isaaclab.sh -p scripts/biped/rsl_rl/benchmark.py --task SteppingStone  --num_envs 5 --use_rl

# with logging
./isaaclab.sh -p scripts/biped/rsl_rl/benchmark.py --task SteppingStone  --num_envs 5 --use_rl --log
```