# Hector IsaacLab simulation 
This is simulation code for "RL-augmented Adaptive Model Predictive Control for Bipedal Locomotion over Challenging Terrain". 
Later, this code will be integrated to the latest IsaacLab. 

## Initial setups

### Install IsaacSim 4.5
See official [docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim) \
Use conda and pip install IsaacSim4.5. 

### Clone this repository
```bash
git clone git@github.com:rl-augmented-mpc/IsaacLab4.5.git -b devel 
```

### Pull usd asset via git LFS
```bash
git lfs pull
```

### Install IsaacLab
```bash
conda activate env_isaaclab
cd ~/IsaacLab
./isaaclab.sh -i
```

### Install C++ MPC Controller
```bash
cd ~
git clone git@github.com:rl-augmented-mpc/HECTOR_HW.git
cd HECTOR_HW
./isaaclab.sh -p -m pip install -e .
```

### Install PyTorch MPC controller
```bash
cd ~
git clone git@github.com:rl-augmented-mpc/Biped-PyMPC.git
cd Biped-PyMPC
./isaaclab.sh -p -m pip install -e .
```

### Install RL libraries
We use modified library of rl_games (recommended). 
```bash
cd ~
git clone git@github.com:rl-augmented-mpc/rl_games.git

# and install
cd ~/rl_games
./isaaclab.sh -p -m pip install -e .
./isaaclab.sh -p -m pip install ray
```

## Run code

### Run MPC

#### w/ C++ MPC
```bash
./isaaclab.sh -p scripts/biped/convex_mpc/play.py --task HECTOR-ManagerBased-RL-SAC-Rough-Blind-PLAY --num_envs 1 --max_trials 1 --episode_length 20
```
#### w/ batch-MPC
```bash
./isaaclab.sh -p scripts/biped/convex_mpc/play.py --task HECTOR-ManagerBased-RL-GPU-SAC-Rough-Blind --num_envs 1024 --max_trials 1 --episode_length 20
```

### Train policy

#### w/ C++ MPC
```bash
./isaaclab.sh -p scripts/biped/rl_games/train.py --task HECTOR-ManagerBased-RL-SAC-Rough-Blind --num_envs 24 --video --headless
```

#### w/ batch-MPC
```bash
./isaaclab.sh -p scripts/biped/rl_games/train.py --task HECTOR-ManagerBased-RL-GPU-SAC-Rough-Blind --num_envs 1024 --video --headless
```

### Run trained policy

#### w/ C++ MPC
```bash
./isaaclab.sh -p scripts/biped/rl_games/play.py --task HECTOR-ManagerBased-RL-SAC-Rough-Blind-PLAY --num_envs 1 --max_trials 1 --use_rl
```

#### w/ batch-MPC
```bash
./isaaclab.sh -p scripts/biped/rl_games/play.py --task HECTOR-ManagerBased-RL-GPU-SAC-Rough-Blind-PLAY --num_envs 1 --max_trials 1 --use_rl
```