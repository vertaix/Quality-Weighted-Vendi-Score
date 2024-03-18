import torch
import gpytorch
import numpy as np
from gpytorch.mlls import PredictiveLogLikelihood 
from robot.trust_region import TrustRegionState, update_state, generate_batch
from robot.gp_utils.update_models import update_surr_model
from robot.gp_utils.ppgpr import GPModelDKL

import sys

sys.path.append("../../diverse_search")

from utils import tiebreak_max, vendi_score


def append_gram_matrix(gram_matrix, objective, ix, selected_xs, this_x):
    if ix == 0:
        return gram_matrix

    dist_vector = []
    for other_ix in range(ix):
        dist = objective.divf(selected_xs[other_ix], this_x)
        dist_vector.append(dist)
    dist_vector = np.array(dist_vector)
    similiarity_vector = np.exp(-dist_vector ** 2 / 2)

    return np.concatenate(
        (
            np.concatenate(
                (gram_matrix, similiarity_vector.reshape(-1, 1)), axis=1
            ),
            np.append(similiarity_vector, [1.]).reshape(1, -1)
        ),
        axis=0,
    )


def sequential_greedy_maximize_qvs(
    candidate_xs, scores, objective, q, q_taus, b, count_mask=None
):
    if count_mask is None:
        count_mask = [True] * len(scores)

    selected_idxs = []
    selected_xs = []
    selected_scores = []
    valid = []

    ix = 0
    count = 0
    while count < b:
        if ix == 0:
            top_idx = torch.argmax(scores)
            gram_matrix = np.ones((1, 1))
            valid.append(True)
        else:
            current_score_sum = sum(selected_scores)
            all_qvs = []

            for candidate_ix in range(len(scores)):
                if candidate_ix in selected_idxs:
                    all_qvs.append(-float("Inf"))
                else:
                    candidate_x = candidate_xs[candidate_ix]
                    candidate_score = scores[candidate_ix].item()

                    candidate_gram_matrix = append_gram_matrix(
                        gram_matrix, objective, ix, selected_xs, candidate_x
                    )

                    vs = vendi_score(candidate_gram_matrix, q=q)
                    avg_score = (current_score_sum + candidate_score) / (ix + 1)
                    qvs = avg_score * vs
                    if vs < q_taus[ix]:
                        qvs = -float("Inf")
                    all_qvs.append(qvs)

            top_idx, _ = tiebreak_max(np.array(all_qvs))
            valid.append(all_qvs[top_idx] != -float("Inf"))
            print(f"{all_qvs[top_idx]=}")
        
        top_score = scores[top_idx].item()
        top_x = candidate_xs[top_idx]

        selected_idxs.append(top_idx)
        selected_xs.append(top_x)
        selected_scores.append(top_score)

        gram_matrix = append_gram_matrix(
            gram_matrix, objective, ix, selected_xs, top_x
        )

        ix += 1
        if count_mask[top_idx]:
            count += 1
        
    print(f"{selected_scores=}")
    print(f"{gram_matrix=}")
    
    return np.array(selected_idxs), np.array(selected_xs), valid


class RobotState:

    def __init__(
        self,
        M,
        tau,
        objective,
        train_x,
        train_y,
        train_z=None,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        mode="threshold",
        q=1.0,
        acq_func='ts',
        verbose=True,
    ):

        self.tau                = tau               # Diversity threshold
        self.M                  = M                 # Number of diverse soltuions we seek
        self.objective          = objective         # Objective with vae and associated diversity function for particular task
        self.train_x            = train_x           # initial train x data
        self.train_y            = train_y           # initial train y data
        self.train_z            = train_z           # initial train z data (for latent space objectives, see lolrobot subclasss)
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.learning_rte       = learning_rte      # lr to use for model updates
        self.bsz                = bsz               # acquisition batch size
        self.mode               = mode
        self.q                  = q
        if self.mode == "qVS":
            covar = np.exp(-tau**2 / 2)
            self.q_taus = []
            for i in range(1, self.M + 2):
                mat = np.ones((i, i), dtype=float) * covar
                mat[np.diag_indices(i)] = 1.
                self.q_taus.append(vendi_score(mat, q=1.))
            print(f"{self.q_taus=}")

        self.acq_func           = acq_func          # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose            = verbose

        assert acq_func == "ts"
        if minimize:
            self.train_y = self.train_y * -1

        self.num_new_points = 0 # number of newly acquired points (in acquisiton)
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs

        self.initialize_global_surrogate_model()
        self.initialize_xs_to_scores_dict()
        self.initialize_tr_states()


    def search_space_data(self):
        return self.train_x

    def initialize_xs_to_scores_dict(self,):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict


    def initialize_tr_states(self):
        self.rank_ordered_trs = [] 
        for _ in range(self.M):
            state = TrustRegionState( # initialize turbo state
                    dim=self.objective.dim,
                    batch_size=self.bsz, 
                    center_point=None,
                )
            self.rank_ordered_trs.append(state)
        
        # find feasible tr centers to start 
        self.recenter_trs() 


    def recenter_trs(self):
        # recenter trust regions and log best diverse set found 
        M_diverse_scores = []
        tr_center_xs = []

        if self.mode == "threshold":  # original ROBOT
            idx_num = 0
            _, top_t_idxs = torch.topk(self.train_y.squeeze(), len(self.train_y))
            for ix, state in enumerate(self.rank_ordered_trs):
                while True: 
                    # if we run out of feasible points in dataset
                    if idx_num >= len(self.train_y): 
                        # Randomly sample a new feasible point (rare occurance)
                        center_x, center_point, center_score = self.randomly_sample_feasible_center(higher_ranked_xs=tr_center_xs) 
                        break
                    # otherwise, finding highest scoring feassible point in remaining dataset for tr center
                    center_idx = top_t_idxs[idx_num]
                    center_score = self.train_y[center_idx].item()
                    center_point = self.search_space_data()[center_idx] 
                    center_x = self.train_x[center_idx]
                    idx_num += 1
                    if self.is_feasible(center_x, higher_ranked_xs=tr_center_xs):
                        break 

                tr_center_xs.append(center_x) 
                M_diverse_scores.append(center_score)
                state.center_point = center_point
                state.best_value = center_score
                state.best_x = center_x 
        elif self.mode in ["TuRBO", "random"]:
            _, top_t_idxs = torch.topk(self.train_y.squeeze(), len(self.train_y))
            for ix, state in enumerate(self.rank_ordered_trs):
                center_idx = top_t_idxs[ix]
                center_score = self.train_y[center_idx].item()
                center_point = self.search_space_data()[center_idx] 
                center_x = self.train_x[center_idx]

                tr_center_xs.append(center_x) 
                M_diverse_scores.append(center_score)
                state.center_point = center_point
                state.best_value = center_score
                state.best_x = center_x
        else:
            assert self.mode == "qVS"

            center_idxs, _, valid = sequential_greedy_maximize_qvs(
                # self.train_x, scores, self.objective, self.q, self.M
                self.train_x, 
                self.train_y.detach().clone(), 
                self.objective, 
                self.q, 
                self.q_taus, 
                self.M,
            )

            for ix, state in enumerate(self.rank_ordered_trs):
                if valid[ix]:
                    tr_center_xs.append(self.train_x[center_idxs[ix]])
                    M_diverse_scores.append(self.train_y[center_idxs[ix]].item())
                    state.center_point = self.search_space_data()[center_idxs[ix]]
                    state.best_value = self.train_y[center_idxs[ix]].item()
                    state.best_x = self.train_x[center_idxs[ix]]
                else:
                    center_x, center_point, center_score = self.randomly_sample_feasible_center(
                        higher_ranked_xs=tr_center_xs
                    ) 
                    tr_center_xs.append(center_x) 
                    M_diverse_scores.append(center_score)
                    state.center_point = center_point
                    state.best_value = center_score
                    state.best_x = center_x

        self.M_diverse_scores = np.array(M_diverse_scores)
        self.M_diverse_xs = tr_center_xs


    def restart_trs_as_needed(self):
        for ix, state in enumerate(self.rank_ordered_trs):
            if state.restart_triggered:
                new_state = TrustRegionState( 
                    dim=self.objective.dim,
                    batch_size=self.bsz, 
                    center_point=state.center_point,
                    best_value=state.best_value,
                    best_x=state.best_x
                )
                self.rank_ordered_trs[ix] = new_state


    def initialize_global_surrogate_model(self ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        n_pts = min(self.search_space_data().shape[0], 1024)
        self.model = GPModelDKL(self.search_space_data()[:n_pts, :], likelihood=likelihood )
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.search_space_data().size(-2))
        self.model = self.model.eval() 
        self.model = self.model


    def update_surrogate_model(self ): 
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            X = self.search_space_data()
            Y = self.train_y.squeeze(-1)
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            X = self.search_space_data()[-self.num_new_points:]
            Y = self.train_y[-self.num_new_points:].squeeze(-1)
            
        self.model = update_surr_model(
            self.model,
            self.mll,
            self.learning_rte,
            X,
            Y,
            n_epochs
        )
        self.initial_model_training_complete = True


    def is_feasible(self, x, higher_ranked_xs): 
        for higher_ranked_x in higher_ranked_xs:
            if self.objective.divf(x, higher_ranked_x) < self.tau:
                return False 
        return True 
    

    def is_feasible_vs(self, x, higher_ranked_xs, higher_ranked_gram_matrix):
        num_higher_ranked_xs = higher_ranked_gram_matrix.shape[-1]

        candidate_gram_matrix = append_gram_matrix(
            higher_ranked_gram_matrix, 
            self.objective, 
            num_higher_ranked_xs,
            higher_ranked_xs,
            x,
        )

        vs = vendi_score(candidate_gram_matrix, q=self.q) 
        
        # normalized_vs = vs / (num_higher_ranked_xs + 1)
        # return normalized_vs > self.q_tau

        # return vs > self.q_tau
        return vs > self.q_taus[num_higher_ranked_xs]


    def sample_random_searchspace_points(self, N):
        lb, ub = self.objective.lb, self.objective.ub 
        if ub is None: ub = self.search_space_data().max() 
        if lb is None: lb = self.search_space_data().max() 
        return torch.rand(N, self.objective.dim)*(ub - lb) + lb


    def randomly_sample_feasible_center(self, higher_ranked_xs, max_n_samples=1_000):
        ''' Rare edge case when we run out of feasible evaluated datapoints
        and must randomly sample to find a new feasible center point
        '''
        n_samples = 0
        while True:
            if n_samples > max_n_samples:
                raise RuntimeError(f'Failed to find a feasible tr center after {n_samples} random samples, recommend use of smaller M or smaller tau')
            center_x = self.sample_random_searchspace_points(N=1)
            n_samples += 1
            if self.is_feasible(center_x, higher_ranked_xs=higher_ranked_xs):
                out_dict = self.objective(center_x)
                center_score = out_dict['scores'].item() 
                # add new point to existing dataset 
                self.update_next(
                    y_next=torch.tensor(center_score).float(),
                    x_next=center_x,
                )
                break 

        return center_x, center_x, center_score 


    def generate_batch_single_tr(self, tr_state):
        search_space_cands = generate_batch(
            state=tr_state,
            model=self.model,
            X=self.search_space_data(),
            Y=self.train_y,
            batch_size=self.bsz, 
            acqf=self.acq_func,
            absolute_bounds=(self.objective.lb, self.objective.ub)
        )

        return search_space_cands


    def remove_infeasible_candidates(self, x_cands, higher_ranked_cands):
        feasible_xs = []
        bool_arr = []
        for x_cand in x_cands:
            test1 = self.is_feasible(x_cand, higher_ranked_cands)
            if test1: # self.is_feasible(x_cand, higher_ranked_cands):
                if type(x_cand) is torch.Tensor:
                    feasible_xs.append(x_cand.unsqueeze(0))
                else:
                    feasible_xs.append(x_cand)
                bool_arr.append(True)
            else:
                bool_arr.append(False)
        # tracks which were removed vs. kept
        bool_arr = np.array(bool_arr)

        return feasible_xs, bool_arr


    def remove_infeasible_candidates_vs(self, x_cands, higher_ranked_cands):
        num_higher_ranked_cands = len(higher_ranked_cands)
        for ix in range(num_higher_ranked_cands):
            if ix == 0:
                higher_ranked_gram_matrix = np.ones((1, 1))
            else:
                higher_ranked_gram_matrix = append_gram_matrix(
                    higher_ranked_gram_matrix, 
                    self.objective, 
                    ix, 
                    higher_ranked_cands,
                    higher_ranked_cands[ix],
                )

        feasible_xs = []
        bool_arr = []
        for x_cand in x_cands:
            test1 = (
                (num_higher_ranked_cands == 0) or
                self.is_feasible_vs(
                    x_cand, higher_ranked_cands, higher_ranked_gram_matrix
                )
            )
            if test1: # self.is_feasible(x_cand, higher_ranked_cands):
                if type(x_cand) is torch.Tensor:
                    feasible_xs.append(x_cand.unsqueeze(0))
                else:
                    feasible_xs.append(x_cand)
                bool_arr.append(True)
            else:
                bool_arr.append(False)
        # tracks which were removed vs. kept
        bool_arr = np.array(bool_arr)

        return feasible_xs, bool_arr


    def update_feasible_candidates_and_tr_state(self, state, feasible_searchspace_pts, feasible_ys ):
        if len(feasible_ys) > 0:
            if type(feasible_searchspace_pts) is np.ndarray:
                feasible_searchspace_pts = torch.from_numpy(feasible_searchspace_pts).float() 
            if self.minimize:
                feasible_ys = feasible_ys * -1
            self.all_feasible_ys = self.all_feasible_ys + feasible_ys.tolist()
            feasible_searchspace_pts = feasible_searchspace_pts.detach().cpu() 
            self.all_feasible_searchspace_pts = torch.cat((self.all_feasible_searchspace_pts, feasible_searchspace_pts))
            # 4. update state of this tr only on the feasible ys it suggested
            update_state(state, feasible_ys)


    def compute_scores_remaining_cands(self, feasible_searchspace_pts):
        out_dict = self.objective(torch.cat(feasible_searchspace_pts) )
        feasible_searchspace_pts=out_dict['valid_xs'] 
        feasible_ys=out_dict['scores']
        
        return feasible_ys, feasible_searchspace_pts


    def compute_scores_and_update_state(self, state, feasible_searchspace_pts):
        if len(feasible_searchspace_pts) > 0:
            # Compute scores for remaining feasible candiates
            feasible_ys, feasible_searchspace_pts = self.compute_scores_remaining_cands(feasible_searchspace_pts)
            # Update tr state on feasible candidates 
            self.update_feasible_candidates_and_tr_state(state, feasible_searchspace_pts, feasible_ys )


    def get_feasible_cands(self, cands):
        if self.mode == "threshold":
            feasible_searchspace_pts, _ = self.remove_infeasible_candidates(
                x_cands=cands, 
                higher_ranked_cands=self.all_feasible_searchspace_pts
            )
        # elif self.mode == "qVS":
        #     feasible_searchspace_pts, _ = self.remove_infeasible_candidates_vs(
        #         x_cands=cands, 
        #         # higher_ranked_cands=self.all_feasible_searchspace_pts
        #         higher_ranked_cands=torch.stack([state.center_point for state in self.rank_ordered_trs])
        #     )
        else:
            assert self.mode in ["qVS", "TuRBO", "random"]

            feasible_searchspace_pts = []
            for cand in cands:
                if type(cand) is torch.Tensor:
                    feasible_searchspace_pts.append(cand.unsqueeze(0))
                else:
                    feasible_searchspace_pts.append(cand)

        return feasible_searchspace_pts


    def asymmetric_acquisition(self):   
        '''Generate new candidate points,
        asymetrically discard infeasible ones, 
        evaluate them, and update data
        '''
        self.all_feasible_xs = [] # (used only by LOL-ROBOT when searchspace pts != xs)
        self.all_feasible_ys = []
        self.all_feasible_searchspace_pts = torch.tensor([])
        for state in self.rank_ordered_trs:
            # 1. Generate a batch of candidates in 
            #   trust region using global surrogate model
            if self.mode == "random":
                lb, ub = self.objective.lb, self.objective.ub 
                x_next = torch.rand(self.bsz, self.objective.dim)*(ub - lb) + lb
            else:
                x_next = self.generate_batch_single_tr(state)

            # 2. Asymetrically remove infeasible candidates
            feasible_searchspace_pts = self.get_feasible_cands(x_next)

            # 3. Compute scores for feassible cands and update tr statee 
            self.compute_scores_and_update_state(state, feasible_searchspace_pts)

        # 4. Add all new evaluated points to dataset (update_next)
        if len(self.all_feasible_searchspace_pts ) != 0:
            self.num_new_points = len(self.all_feasible_ys)
        self.update_data_all_feasible_points() 


    def update_data_all_feasible_points(self):
        if len(self.all_feasible_searchspace_pts ) != 0:
            self.update_next(
                y_next_=torch.tensor(self.all_feasible_ys).float(),
                x_next_=self.all_feasible_searchspace_pts,
            )


    def update_next(self, y_next_, x_next_):
        '''Add new points (y_next, x_next) to train data
        '''
        x_next_ = x_next_.detach().cpu() 
        if len(x_next_.shape) == 1:
            x_next_ = x_next_.unsqueeze(0)
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        #if we imporve 
        if y_next_.max() > self.best_score_seen:
            self.best_score_seen = y_next_.max().item() 
            self.best_x_seen = x_next_[y_next_.argmax()] 
        y_next_ = y_next_.unsqueeze(-1)
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
        self.train_x = torch.cat((self.train_x, x_next_), dim=-2)

        print(f"{self.train_y.shape=}")
        print(f"{self.train_y.max().item()=}")
