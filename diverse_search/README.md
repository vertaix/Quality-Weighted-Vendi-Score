## Weights and Biases (wandb) tracking
This repo is set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site.
By default, the code is run without wandb tracking. After creating an account, wandb tracking can be used for optimization runs by simply adding the following args `--track_with_wandb True --wandb_entity nmaus` (see example commands below). 

## Getting the data

The data for the photoswitch and bulk metal glass search problems used in our paper can be downloaded from Box via this link: [https://wustl.box.com/s/cxxxw7za5i7c782aa8xhumq4e2067ha8](https://wustl.box.com/s/cxxxw7za5i7c782aa8xhumq4e2067ha8).
The downloaded `data` folder should be placed in the root directory of the repository.

## Running

Run `batch_run.py` with desired command line arguments.

```Bash
python3 batch_run.py --datapath ../data/[DATA] --strategy VAS --budget 100 --batch_size 5
```
where `[DATA]` should be either `photoswitch` or `bmg`.

## Analyses

Jupyter notebooks that perform analyses of search results can be found in the `examples` folder in the root directory.
