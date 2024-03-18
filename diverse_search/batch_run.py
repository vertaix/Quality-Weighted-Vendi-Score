import os
import argparse
from time import time

import numpy as np

from utils import KNNModel, Experiment
from batch_policies import *

import wandb


def run_as_loop(
    datapath, 
    budget, 
    batch_size,
    strategy, 
    seed=0, 
    q="1",
    continue_run=False, 
    continue_run_id=None, 
    **kwargs,
):
    np.random.seed(seed)
    if not q == "inf":
        q = float(q)

    dataname = os.path.basename(os.path.normpath(datapath))

    exp = Experiment(datapath, budget, batch_size)

    prevalence = exp.labels.sum() / exp.labels.size
    alpha = [1 - np.round(prevalence, 3), np.round(prevalence, 3)]

    model = KNNModel(alpha, exp.csr_weights)

    if strategy == "random":
        policy = Random(batch_size)
        strategy_name = strategy
    if strategy == "ECI":
        policy = BatchExpectedCoverageImprovement(
            model,
            kwargs["threshold"],
            exp.similarities,
            exp.nearest_neighbors,
            batch_size,
            kwargs["fantasy_oracle"],
        )
        strategy_name = f"{kwargs['fantasy_oracle']}_{kwargs['threshold']:.2f}_{strategy}"
    if strategy == "SELECT":
        policy = BatchSELECT(
            model, 
            kwargs["beta"], 
            kwargs["tradeoff_lambda"], 
            batch_size, 
            kwargs["fantasy_oracle"],
        )
        strategy_name = f"{kwargs['fantasy_oracle']}_{kwargs['beta']:.2f}_{kwargs['tradeoff_lambda']}_{strategy}"
    if strategy == "one-step":
        policy = BatchOneStepActiveSearch(model, batch_size)
        strategy_name = strategy
    elif strategy == "ENS":
        policy = BatchEfficientNonmyopicSearch(
            model, batch_size, kwargs["fantasy_oracle"]
        )
        strategy_name = f"{kwargs['fantasy_oracle']}_{strategy}"
    elif strategy == "VAS":
        policy = BatchVendiActiveSearch(
            model, exp.csr_weights, batch_size, q
        )
        strategy_name = strategy
    
    if continue_run:
        # read and process past queries
        api = wandb.Api()
        crashed_run = api.run(f"quanng/diverse-search/{continue_run_id}")
        queried_ind = crashed_run.history()["next_ind"].values

        print("loading past indices:", queried_ind)

        for ind in queried_ind:
            exp.evaluate(ind)
        
        # resume the run
        wandb_run = wandb.init(
            project="diverse-search", id=continue_run_id, resume="allow"
        )
    else:
        wandb_run = wandb.init(
            project="diverse-search", 
            entity="quanng",
            config={"seed": seed}
        )
        wandb_run.name = f"{dataname}_{q}_{budget}_{batch_size}_{strategy_name}"

    while exp.budget > 0:
        print("remaining budget:", exp.budget)

        t0 = time()

        if strategy in ["random", "ECI", "SELECT", "one-step", "VAS"]:
            next_ind = policy.get_next_batch(
                exp.test_ind, exp.train_ind, exp.observed_labels
            )
        elif strategy == "ENS":
            next_ind = policy.get_next_batch(
                exp.test_ind, exp.train_ind, exp.observed_labels, exp.budget
            )
        
        duration = time() - t0

        exp.evaluate(next_ind)
        
        next_label = exp.observed_labels[-batch_size:]
        linear_utility = exp.observed_labels.sum()
        pos_ind = exp.train_ind[exp.observed_labels > 0]
        vs_utility = vendi_score(
            sub_sparse_matrix(exp.symmetric_csr_weights, pos_ind, pos_ind),
            q=q,
        )
        
        for batch_i in range(batch_size):
            wandb_run.log(
                {
                    "next_ind": next_ind[batch_i],
                    "label": next_label[batch_i],
                    "iteration": exp.train_ind.size - batch_size + batch_i,
                }
            )
        
        wandb_run.log(
            {
                "duration": duration,
                "linear_utility": linear_utility,
                "vs_utility": vs_utility,
                "iteration": exp.train_ind.size,
            }
        )
    
    wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vendi AS")
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--q", default="1", type=str)
    parser.add_argument(
        "--strategy", 
        default="ECI", 
        type=str, 
        choices=["random", "ECI", "SELECT", "one-step", "ENS", "VAS"],
    )
    parser.add_argument("--threshold", default=0.25, type=float)
    parser.add_argument("--beta", default=1., type=float)
    parser.add_argument("--tradeoff_lambda", default=0.25, type=float)
    parser.add_argument("--fantasy_oracle", default="pessimistic", type=str)
    parser.add_argument("--continue_run", type=bool, default=False)
    parser.add_argument("--continue_run_id", type=str, default="")

    args = parser.parse_args()

    run_as_loop(
        args.datapath, 
        args.budget, 
        args.batch_size,
        args.strategy, 
        args.seed, 
        args.q,
        args.continue_run,
        args.continue_run_id,
        threshold=args.threshold,
        beta=args.beta,
        tradeoff_lambda=args.tradeoff_lambda,
        fantasy_oracle=args.fantasy_oracle,
    )
