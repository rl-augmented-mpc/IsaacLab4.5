## Hector IsaacLab simulation 

## Before begin

### Install IsaacSim 4.5
See this [docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim) \
Use conda and pip install of IsaacSim4.5. 

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
git clone https://github.gatech.edu/GeorgiaTechLIDARGroup/HECTOR_HW_new.git -b feature/gait_param_reset_code
./isaaclab.sh -p -m pip install -e {/path/to/hector/controller}
```

## Run code

### Run MPC

Only MPC

```bash
./isaaclab.sh -p scripts/biped/run_direct.py --task Hector-Hierarchical-Prime-Rigid --num_envs 1 --max_trials 10 --episode_length 20
```

### Train RL 
This is only available for direct based 
```bash
./isaaclab.sh -p scripts/biped/rsl/train.py --task Hector-Hierarchical-Prime-Rigid --num_envs 64 --videos --headless
```
After a while, you will see the logs under `logs` directory (for example `logs/rsl_rl/ppo_rsl_rl_lstm_friction/2025-03-17_15-31-27`). 

### Run trained policy 
This is only available for direct based 
```bash
./isaaclab.sh -p scripts/biped/rsl/benchmark.py --task Hector-Hierarchical-Prime-Rigid --num_envs 5 --use_rl

# with logging
./isaaclab.sh -p scripts/biped/rsl/benchmark.py --task Hector-Hierarchical-Prime-Rigid --num_envs 5 --use_rl --log
```