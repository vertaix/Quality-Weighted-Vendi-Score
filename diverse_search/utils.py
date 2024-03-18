import os

import numpy as np
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix, load_npz, eye


def tiebreak_max(scores):
    max_score = scores.max()
    tie_places = np.where(scores == max_score)[0]
    
    selected_place = np.random.choice(tie_places)
    
    return selected_place, max_score


def get_fantasy_label(fantasy_oracle):
    if fantasy_oracle == "pessimistic":
        return 0
    elif fantasy_oracle == "optimistic":
        return 1

    raise ValueError("invalid valud for fantasy_oracle")

def entropy_q(p, q=1.):
    p_ = p[p > 0]
    if q == 1.:
        return np.sum(-(p_ * np.log(p_)))
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_ ** q).sum()) / (1 - q)


def vendi_score(K, q=1.):
    lambdas, _ = np.linalg.eigh(K)
    lambdas[lambdas <= 0] = 0
    normalized_lambdas = lambdas / np.sum(lambdas)
    
    entropy = entropy_q(normalized_lambdas, q=q)
    return np.exp(entropy)


def sub_sparse_matrix(matrix, row, col, toarray=True):
    submatrix = matrix[row].tocsc()[:, col]
    if toarray:
        return submatrix.toarray()
    return submatrix


class KNNModel:
    def __init__(self, alpha: np.array, weights: csr_matrix):
        self._alpha = alpha
        self._weights = weights

    @property
    def alpha(self) -> np.array:
        return self._alpha

    @property
    def weights(self) -> csr_matrix:
        return self._weights

    def predict(self, test_ind: np.array, train_ind: np.array, observed_labels: np.array) -> np.array:
        probs = np.empty((test_ind.size, 2))

        pos_ind = (observed_labels > 0)
        neg_ind = ~pos_ind
        masks = [neg_ind, pos_ind]

        csc_weights = self._weights[test_ind].tocsc()

        for c in range(2):
            probs[:, c] = self._alpha[c] + (
                csc_weights[:, train_ind[masks[c]]]
                .sum(axis=1).flatten()
            )

        return normalize(probs, axis=1, norm="l1")[:, 1]


class Experiment:
    def __init__(self, datapath, budget, batch_size):
        self.features = np.load(os.path.join(datapath, "features.npy"))
        self.labels = np.load(os.path.join(datapath, "labels.npy"))

        self.n = self.labels.size

        self.nearest_neighbors = np.load(os.path.join(datapath, "nearest_neighbors.npy"))
        self.similarities = np.load(os.path.join(datapath, "similarities.npy"))

        self.csr_weights = load_npz(os.path.join(datapath, "csr_weights.npz"))
        self.symmetric_csr_weights = (self.csr_weights + self.csr_weights.T) / 2

        self.budget = budget
        self.batch_size = batch_size

        self.train_ind = np.array(
            [np.random.choice(np.where(self.labels > 0)[0])]
        )
        self.observed_labels = self.labels[self.train_ind]
        self.test_ind = np.delete(np.arange(self.n), self.train_ind)
    
    def evaluate(self, next_ind):
        assert self.budget >= self.batch_size
        assert next_ind.size == self.batch_size

        for ind in next_ind:
            assert ind not in self.train_ind

        self.train_ind = np.append(self.train_ind, next_ind)
        self.observed_labels = self.labels[self.train_ind]
        
        self.budget -= next_ind.size
        self.test_ind = np.delete(np.arange(self.n), self.train_ind)
