import os
import argparse
from time import time

import numpy as np

from utils import KNNModel, Experiment
from policies import *

import wandb


def run_as_loop(
    datapath, 
    budget, 
    strategy, 
    seed=0,
    q=1, 
    continue_run=False, 
    continue_run_id=None, 
    **kwargs
):
    np.random.seed(seed)

    dataname = os.path.basename(os.path.normpath(datapath))

    exp = Experiment(datapath, budget)

    prevalence = exp.labels.sum() / exp.labels.size
    alpha = [1 - np.round(prevalence, 3), np.round(prevalence, 3)]

    model = KNNModel(alpha, exp.csr_weights)

    if strategy == "CAS":
        policy = ExpectedCoverageImprovement(
            model,
            kwargs["threshold"],
            exp.similarities,
            exp.nearest_neighbors,
        )
    elif strategy == "ENS":
        policy = EfficientNonmyopicSearch(model)
    elif strategy == "vENS":
        policy = VendiEfficientNonmyopicSearch(
            model, exp.csr_weights, kwargs["compute_p"]
        )
    
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
        wandb_run = wandb.init(project="diverse-search", entity="quanng")
        wandb_run.name = f"{dataname}_{q}_{budget}_{strategy}"

    while exp.budget > 0:
        print("remaining budget:", exp.budget)

        t0 = time()

        if strategy == "CAS":
            scores = policy.get_scores(
                exp.test_ind, exp.train_ind, exp.observed_labels
            )
        elif strategy in ["ENS", "vENS"]:
            scores = policy.get_scores(
                exp.test_ind, exp.train_ind, exp.observed_labels, exp.budget
            )
        
        duration = time() - t0

        selected_place, _ = tiebreak_max(scores)
        next_ind = exp.test_ind[selected_place]
        exp.evaluate(next_ind)
        
        next_label = exp.observed_labels[-1]
        linear_utility = exp.observed_labels.sum()
        pos_ind = exp.train_ind[exp.observed_labels > 0]
        vs_utility = vendi_score(
            sub_sparse_matrix(exp.symmetric_csr_weights, pos_ind, pos_ind),
            q=self.q,
        )
        
        wandb_run.log(
            {
                "seed": seed,
                "next_ind": next_ind,
                "label": next_label,
                "duration": duration,
                "linear_utility": linear_utility,
                "vs_utility": vs_utility,
            }
        )
    
    wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vendi AS")
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--q", default=1., type=float)
    parser.add_argument("--strategy", default="CAS", type=str, choices=["CAS", "ENS", "vENS"])
    parser.add_argument("--threshold", default=0.25, type=float)
    parser.add_argument("--compute_p", default=1., type=float)
    parser.add_argument("--continue_run", type=bool, default=False)
    parser.add_argument("--continue_run_id", type=str, default="")

    args = parser.parse_args()

    run_as_loop(
        args.datapath, 
        args.budget, 
        args.strategy, 
        args.seed, 
        args.q,
        args.continue_run,
        args.continue_run_id,
        threshold=args.threshold,
        compute_p=args.compute_p,
    )
