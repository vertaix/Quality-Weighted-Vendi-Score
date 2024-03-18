import numpy as np
from scipy.sparse import eye

from utils import tiebreak_max, sub_sparse_matrix, vendi_score


class ExpectedCoverageImprovement:
    def __init__(self, model, threshold, similarities, nearest_neighbors):
        self.model = model
        self.threshold = threshold
        
        self.similarities = similarities
        self.nearest_neighbors = nearest_neighbors
        
        self.n = similarities.shape[0]
        self.k = similarities.shape[1]
    
    def get_scores(self, test_ind, train_ind, observed_labels):
        probs = self.model.predict(test_ind, train_ind, observed_labels)
        
        reverse_ind = np.zeros(self.n, dtype=int) * -1
        reverse_ind[test_ind] = np.arange(test_ind.size)
        
        # block out points that are too close to labeled data
        for _, this_train_ind in enumerate(train_ind):
            cutoff = self.k - np.searchsorted(
                self.similarities[this_train_ind, :][::-1], 
                self.threshold, 
                side="right",
            )

            if cutoff > 0:
                covered_nearest_neighbors = self.nearest_neighbors[this_train_ind, :cutoff]
                covered_nearest_neighbors_ind = reverse_ind[covered_nearest_neighbors]
                assert -1 not in covered_nearest_neighbors_ind

                probs[covered_nearest_neighbors_ind] = 0.
        
        # compute score
        scores = np.zeros_like(test_ind, dtype=float)

        for test_ind_i, this_test_ind in enumerate(test_ind):
            cutoff = self.k - np.searchsorted(
                self.similarities[this_test_ind, :][::-1], 
                self.threshold, 
                side="right",
            )

            if cutoff > 0:
                covered_nearest_neighbors = self.nearest_neighbors[this_test_ind, :cutoff]
                covered_nearest_neighbors_ind = reverse_ind[covered_nearest_neighbors]
                assert -1 not in covered_nearest_neighbors_ind

                scores[test_ind_i] = probs[covered_nearest_neighbors_ind].sum()
                
        return scores


class EfficientNonmyopicSearch:
    def __init__(self, model):
        self.model = model
    
    def get_scores(self, test_ind, train_ind, observed_labels, budget):
        probs = self.model.predict(test_ind, train_ind, observed_labels)
        if budget == 1:
            return probs
        
        k = budget - 1
        
        # compute score
        scores = np.zeros_like(test_ind, dtype=float)

        for test_ind_i, this_test_ind in enumerate(test_ind):
            fake_train_ind = np.append(train_ind, this_test_ind)
            fake_test_ind = np.delete(test_ind, test_ind_i)
            
            fake_utilities = np.empty(2)
            for fake_label in range(2):
                fake_observed_labels = np.append(observed_labels, fake_label)
                fake_probs = self.model.predict(
                    fake_test_ind, fake_train_ind, fake_observed_labels
                )
                top_ind = np.argpartition(fake_probs, -k)[-k:]
                
                fake_utilities[fake_label] = fake_probs[top_ind].sum()
            
            p = probs[test_ind_i]
            scores[test_ind_i] = p * (fake_utilities[1] + 1) + (1 - p) * fake_utilities[0]
                
        return scores


class VendiEfficientNonmyopicSearch:
    def __init__(self, model, csr_weights, q=1, compute_p=1.):
        self.model = model
        
        self.csr_weights = csr_weights
        self.n = csr_weights.shape[0]
        
        self.symmetric_csr_weights = (csr_weights + csr_weights.T) / 2
        self.zero_diag_csr_weights = csr_weights - eye(self.n)
        self.max_num_influences = (self.zero_diag_csr_weights > 0).sum(axis=0).max()

        self.q = q
        self.compute_p = compute_p
    
    def get_scores(self, test_ind, train_ind, observed_labels, budget):
        def get_qvs_batch(test_ind, train_ind, observed_labels, batch_size):
            probs = self.model.predict(test_ind, train_ind, observed_labels)

            remaining_probs = probs.copy()
            remaining_candidates = test_ind.copy()

            positive_ind = train_ind[observed_labels > 0]
            running_selected_ind = positive_ind.copy()
            current_vs = vendi_score(
                sub_sparse_matrix(
                    self.symmetric_csr_weights, 
                    running_selected_ind, 
                    running_selected_ind,
                ),
                q=self.q,
            )
            current_multiplier = positive_ind.size
            best_qvs = current_vs

            for batch_i in range(batch_size):
                all_vs = np.zeros_like(remaining_candidates, dtype=float)
                all_qvs = np.zeros_like(remaining_candidates, dtype=float)

                prior_mask = sub_sparse_matrix(
                    self.csr_weights, 
                    remaining_candidates, 
                    running_selected_ind, 
                    toarray=False,
                ).sum(axis=1) == 0

                prior_computed = False
                prior_vs = None
                prior_qvs = None

                for candidate_i, candidate_ind in enumerate(remaining_candidates):
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
                            (current_multiplier + remaining_probs[candidate_i])
                            / (current_multiplier + batch_i + 1)
                            * vs
                        )

                        all_vs[candidate_i] = vs
                        all_qvs[candidate_i] = qvs

                        if prior_mask[candidate_i]:
                            prior_vs = vs
                            prior_qvs = qvs
                            prior_computed = True

                selected_place, best_qvs = tiebreak_max(all_qvs)

                running_selected_ind = np.append(
                    running_selected_ind, remaining_candidates[selected_place]
                )

                current_vs = all_vs[selected_place]
                current_multiplier += remaining_probs[selected_place]

                remaining_probs = np.delete(remaining_probs, selected_place)
                remaining_candidates = np.delete(remaining_candidates, selected_place)

            return running_selected_ind[positive_ind.size:], best_qvs 
        
        unlabeled_csc_weights = self.zero_diag_csr_weights[test_ind].tocsc()
        batch_size = budget - 1
        
        nonadaptive_batch, _ = get_qvs_batch(
            test_ind, 
            train_ind, 
            observed_labels, 
            batch_size + self.max_num_influences,
        )
        
        probs = self.model.predict(test_ind, train_ind, observed_labels)
        scores = np.zeros_like(test_ind, dtype=float)
        
        for test_ind_i, this_test_ind in enumerate(test_ind):
            if np.random.rand() < self.compute_p:  # compute with probability p
                fake_train_ind = np.append(train_ind, this_test_ind)
                influenced_ind = test_ind[unlabeled_csc_weights[:, this_test_ind].nonzero()[0]]
                
                fake_test_ind = np.append(influenced_ind, nonadaptive_batch)
                
                fake_utilities = np.empty(2)
                for fake_label in range(2):
                    fake_observed_labels = np.append(observed_labels, fake_label)
                    _, fake_best_qvs = get_qvs_batch(
                        fake_test_ind,
                        fake_train_ind,
                        fake_observed_labels,
                        batch_size,
                    )

                    fake_utilities[fake_label] = fake_best_qvs
                
                p = probs[test_ind_i]
                scores[test_ind_i] = (
                    p * fake_utilities[1] 
                    + (1 - p) * fake_utilities[0]
                )
            else:
                scores[test_ind_i] = -float("inf")
        
        return scores


class SELECT:
    def __init__(self, model, beta, tradeoff_lambda):
        self.model = model
        self.beta = beta
        self.tradeoff_lambda = tradeoff_lambda

    def get_scores(self, test_ind, train_ind, observed_labels):
        probs = self.model.predict(test_ind, train_ind, observed_labels)
        variances = probs * (1 - probs)
        ucb = probs + self.beta * np.sqrt(variances)

        return (
            (1 - self.tradeoff_lambda) * ucb 
            + self.tradeoff_lambda / 2 * np.log(1 + variances)
        )
