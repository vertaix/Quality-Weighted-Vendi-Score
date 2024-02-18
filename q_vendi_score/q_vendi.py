import numpy as np

from vendi_score import vendi


def score(samples, k, s, q=1, p=None, normalize=False):
    quality_scores = [s(sample) for sample in samples]
    quality_mean = np.mean(quality_scores)
    vendi_score = vendi.score(samples, k, q=q, p=p, normalize=normalize)

    return quality_mean * vendi_score


def sequential_maximize_score(
    samples, k, s, target_size, q=1, p=None, normalize=False
):
    if not isinstance(samples, list):
        samples = [sample for sample in samples]
    selected_samples = []
    
    while len(selected_samples) < target_size:
        best_qVS = 0
        for sample_i, sample in enumerate(samples):
            this_qVS = score(
                selected_samples + [sample], 
                k, 
                s, 
                q=q, 
                p=p, 
                normalize=normalize,
            )

            if this_qVS > best_qVS:
                next_sample = sample
                next_sample_i = sample_i
                best_qVS = this_qVS
        
        selected_samples.append(next_sample)
        samples.pop(next_sample_i)

    return selected_samples, this_qVS
