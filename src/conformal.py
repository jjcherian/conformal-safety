import numpy as np

from typing import Callable, List

def compute_conformity_scores(
    dataset : List,
    scores_list : List,
):
    annotations_list = [
        np.asarray([c['is_supported'] for c in unit['atomic_facts']])
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
            c for c, s in zip(unit['atomic_facts'], scores) if s >= t
        ]
        unit['filtered_claims'] = filtered_claims
    return dataset


def assess_factscore_coverage(
    dataset : List,
    nominal_alpha : float
) -> None:
    nonfactual_list = []
    nonfactual_grps = {}
    for d in dataset:
        nonfactual = 'F' in [c['is_supported'] for c in d['filtered_claims']]
        nonfactual_list.append(nonfactual)

        # right now metadata is only *two* strings...TODO this needs to be more flexible
        # if tuple(d['metadata']) not in nonfactual_grps:
        #     nonfactual_grps[tuple(d['metadata'])] = [nonfactual]
        # else:
        #     nonfactual_grps[tuple(d['metadata'])].append(nonfactual)
        # if d['metadata'][0] not in nonfactual_grps:
        #     nonfactual_grps[d['metadata'][0]] = [nonfactual]
        # else:
        #     nonfactual_grps[d['metadata'][0]].append(nonfactual)
        # if d['metadata'][1] not in nonfactual_grps:
        #     nonfactual_grps[d['metadata'][1]] = [nonfactual]
        # else:
        #     nonfactual_grps[d['metadata'][1]].append(nonfactual)
    print(f"Nominal coverage: {nominal_alpha}")
    print(f"Realized marginal coverage: {np.mean(nonfactual_list)}")
    # for grp, nonfactuals in nonfactual_grps.items():
    #     print(f"Realized {grp} coverage: {np.mean(nonfactuals)}")
