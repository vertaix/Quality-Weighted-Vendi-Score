import numpy as np

from policies import *
from utils import tiebreak_max, get_fantasy_label, vendi_score, sub_sparse_matrix


class Random:
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def get_next_batch(self, test_ind, train_ind, observed_labels):
        return np.random.choice(test_ind, size=self.batch_size, replace=False)


class BatchExpectedCoverageImprovement:
    def __init__(
        self, 
        model, 
        threshold, 
        similarities, 
        nearest_neighbors, 
        batch_size,
        fantasy_oracle,
    ):
        self.sequential_policy = ExpectedCoverageImprovement(
            model, threshold, similarities, nearest_neighbors
        )
        
        self.batch_size = batch_size

        assert fantasy_oracle in ["pessimistic", "optimistic"]
        self.fantasy_oracle = fantasy_oracle
    
    def get_next_batch(self, test_ind, train_ind, observed_labels):
        batch_ind = np.empty(self.batch_size, dtype=int)

        fantasy_test_ind = test_ind.copy()
        fantasy_train_ind = train_ind.copy()
        fantasy_observed_labels = observed_labels.copy()
        for batch_i in range(self.batch_size):
            scores = self.sequential_policy.get_scores(
                fantasy_test_ind, fantasy_train_ind, fantasy_observed_labels
            )

            selected_place, _ = tiebreak_max(scores)
            next_ind = fantasy_test_ind[selected_place]

            batch_ind[batch_i] = next_ind
            fantasy_test_ind = np.delete(fantasy_test_ind, selected_place)
            fantasy_train_ind = np.append(fantasy_train_ind, next_ind)
            next_label = get_fantasy_label(self.fantasy_oracle)
            fantasy_observed_labels = np.append(fantasy_observed_labels, next_label)
        
        return batch_ind


class BatchOneStepActiveSearch:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
    
    def get_next_batch(self, test_ind, train_ind, observed_labels):
        probs = self.model.predict(test_ind, train_ind, observed_labels)
        
        selected_place = np.argpartition(
            probs, -self.batch_size
        )[-self.batch_size:]

        return test_ind[selected_place]


class BatchEfficientNonmyopicSearch:
    def __init__(self, model, batch_size, fantasy_oracle):
        self.sequential_policy = EfficientNonmyopicSearch(model)

        self.batch_size = batch_size

        assert fantasy_oracle in ["pessimistic", "optimistic"]
        self.fantasy_oracle = fantasy_oracle
    
    def get_next_batch(self, test_ind, train_ind, observed_labels, budget):
        remain_budget = budget - self.batch_size

        # greedy batch if this is the last batch
        if remain_budget <= 0:
            probs = self.sequential_policy.model.predict(
                test_ind, train_ind, observed_labels
            )
            selected_place = np.argpartition(
                probs, -self.batch_size
            )[-self.batch_size:]

            return test_ind[selected_place]

        batch_ind = np.empty(self.batch_size, dtype=int)

        fantasy_test_ind = test_ind.copy()
        fantasy_train_ind = train_ind.copy()
        fantasy_observed_labels = observed_labels.copy()
        for batch_i in range(self.batch_size):
            scores = self.sequential_policy.get_scores(
                fantasy_test_ind, 
                fantasy_train_ind, 
                fantasy_observed_labels, 
                remain_budget,
            )

            selected_place, _ = tiebreak_max(scores)
            next_ind = fantasy_test_ind[selected_place]

            batch_ind[batch_i] = next_ind
            fantasy_test_ind = np.delete(fantasy_test_ind, selected_place)
            fantasy_train_ind = np.append(fantasy_train_ind, next_ind)
            next_label = get_fantasy_label(self.fantasy_oracle)
            fantasy_observed_labels = np.append(fantasy_observed_labels, next_label)
        
        return batch_ind


class BatchVendiActiveSearch:
    def __init__(self, model, csr_weights, batch_size, q=1):
        self.model = model
        
        self.csr_weights = csr_weights
        self.n = csr_weights.shape[0]
        self.symmetric_csr_weights = (csr_weights + csr_weights.T) / 2

        self.batch_size = batch_size
        self.q = q
    
    def get_next_batch(self, test_ind, train_ind, observed_labels):
        probs = self.model.predict(test_ind, train_ind, observed_labels)

        # greedy search if q = 0
        if self.q == 0:
            selected_place = np.argpartition(
                probs, -self.batch_size
            )[-self.batch_size:]

            return test_ind[selected_place]

        remain_probs = probs.copy()
        remain_candidates = test_ind.copy()

        positive_ind = train_ind[observed_labels > 0]
        running_selected_ind = positive_ind.copy()
        current_multiplier = positive_ind.size
        
        for batch_i in range(self.batch_size):
            all_vs = np.zeros_like(remain_candidates, dtype=float)
            all_qvs = np.zeros_like(remain_candidates, dtype=float)

            prior_mask = sub_sparse_matrix(
                self.csr_weights,
                remain_candidates,
                running_selected_ind,
                toarray=False,
            ).sum(axis=1) == 0

            prior_computed = False
            prior_vs = None
            prior_qvs = None

            for candidate_i, candidate_ind in enumerate(remain_candidates):
                if prior_mask[candidate_i] and prior_computed:
                    all_vs[candidate_i] = prior_vs
                    all_qvs[candidate_i] = prior_qvs
                else:
                    appended_ind = np.append(running_selected_ind, candidate_ind)
                    gram_matrix = sub_sparse_matrix(
                        self.symmetric_csr_weights,
                        appended_ind,
                        appended_ind,
                    )
                    vs = vendi_score(gram_matrix, q=self.q)
                    qvs = (
                        (current_multiplier + remain_probs[candidate_i])
                        / (current_multiplier + batch_i + 1)
                        * vs
                    )

                    all_vs[candidate_i] = vs
                    all_qvs[candidate_i] = qvs

                    if prior_mask[candidate_i]:
                        prior_vs = vs
                        prior_qvs = qvs
                        prior_computed = True
                
            selected_place, _ = tiebreak_max(all_qvs)

            running_selected_ind = np.append(
                running_selected_ind, remain_candidates[selected_place]
            )

            current_multiplier += remain_probs[selected_place]

            remain_probs = np.delete(remain_probs, selected_place)
            remain_candidates = np.delete(remain_candidates, selected_place)
        
        return running_selected_ind[positive_ind.size:]


class BatchSELECT:
    def __init__(self, model, beta, tradeoff_lambda, batch_size, fantasy_oracle):
        self.sequential_policy = SELECT(model, beta, tradeoff_lambda)
        
        self.batch_size = batch_size

        assert fantasy_oracle in ["pessimistic", "optimistic"]
        self.fantasy_oracle = fantasy_oracle
    
    def get_next_batch(self, test_ind, train_ind, observed_labels):
        batch_ind = np.empty(self.batch_size, dtype=int)

        fantasy_test_ind = test_ind.copy()
        fantasy_train_ind = train_ind.copy()
        fantasy_observed_labels = observed_labels.copy()
        for batch_i in range(self.batch_size):
            scores = self.sequential_policy.get_scores(
                fantasy_test_ind, fantasy_train_ind, fantasy_observed_labels
            )

            selected_place, _ = tiebreak_max(scores)
            next_ind = fantasy_test_ind[selected_place]

            batch_ind[batch_i] = next_ind
            fantasy_test_ind = np.delete(fantasy_test_ind, selected_place)
            fantasy_train_ind = np.append(fantasy_train_ind, next_ind)
            next_label = get_fantasy_label(self.fantasy_oracle)
            fantasy_observed_labels = np.append(fantasy_observed_labels, next_label)
        
        return batch_ind
