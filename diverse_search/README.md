## Weights and Biases (wandb) tracking
This repo is set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
By default, the code is run without wandb tracking. After creating an account, wandb tracking can be used for optimization runs by simply adding the following args `--track_with_wandb True --wandb_entity nmaus` (see example commands below). 

## Running

Run `batch_run.py` with desired command line arguments.
We have provided data for the photoswitch search task.

```Bash
python3 batch_run.py --datapath ../data/diverse_search/photoswitch --strategy VAS --budget 100 --batch_size 5
```
