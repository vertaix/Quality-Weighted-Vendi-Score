## Weights and Biases (wandb) tracking
This repo is set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
By default, the code is run without wandb tracking. After creating an account, wandb tracking can be used for optimization runs by simply adding the following args `--track_with_wandb True --wandb_entity nmaus` (see example commands below). 

## Getting Started

### Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

### Environment Setup (Docker)
All optimization tasks can be run in the docker envioronment defined in docker/Dockerfile. Build the docker env using the following steps. 

1. If you do not have a docker account, create one here:
https://hub.docker.com/signup

2. If you would like to use wandb tracking, add your wandb API key to the docker file by adding the following line to docker/Dockerfile: 

```Bash
ENV WANDB_API_KEY=$YOUR_WANDB_API_KEY
```

3. Build the docker file: 

```Bash
cd docker 
docker build -t $YOUR_DOCKER_USER_NAME/robot .
```

The resultant docker environment will have all imports necessary to run ROBOT on all tasks from the paper.

## Running

Run `scripts/continuous_space_optimization.py` with desired command line arguments.
To get a list of command line args specifically for the continuous space optimization tasks, run the following: 

```Bash
cd scripts/
python3 continuous_space_optimization.py -- --help
```

For a list of all remaining possible args that are the more general ROBOT args (not specific to task) run the following:

```Bash
cd scripts/
python3 optimize.py -- --help
```

### Task IDs
The code provides support for three other continuous space optimization tasks. These are the three continuous optimization tasks the original ROBOT paper (see paper for more detialed descriptions of each). 

| task_id | Full Task Name     |
|---------|--------------------|
|  rover  | Rover Trajectory Optimization       |
|  lunar  | Lunar Lander Policy Optimization    |
|  stocks | S&P500 Stock Portfolio Optimization |
