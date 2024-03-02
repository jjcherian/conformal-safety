import numpy as np

from typing import Callable, List

def compute_conformity_scores(
    dataset : List,
    scores_list : List,
):
    annotations_list = [
        np.asarray([True if c.get('annotation', 'F') in ('T', 'S') else False for c in unit['claims']])
        for unit in dataset
    ]
    conf_scores = [np.max(scores[~annotes], initial=0) for scores, annotes in zip(scores_list, annotations_list)]
    return conf_scores

def calibrate_thresholds(
    feats_test : List,
    feats_valid : List,
    scores_valid : List,
    alpha_fn : Callable
) -> List[float]:
    alpha_valid = alpha_fn(feats_valid)
    quantile = np.ceil((1 - alpha_valid[0]) * (len(feats_valid) + 1)) / len(feats_valid)
    return [np.quantile(
        scores_valid,
        q=quantile
    )] * len(feats_test)
    
def conformal_filter(
    dataset : List,
    scores_list : List,
    thresholds : List
) -> List:
    for unit, scores, t in zip(dataset, scores_list, thresholds):
        filtered_claims = [
            c for c, s in zip(unit['claims'], scores) if s >= t
        ]
        unit['filtered_claims'] = filtered_claims
    return dataset