import argparse
import numpy as np

from config import get_config
from conformal import compute_conformity_scores, calibrate_thresholds, conformal_filter
from dataset import load_dataset, split_dataset
from llm_utils import merge_claims


def parse_args():
    parser = argparse.ArgumentParser(
        prog="conformal-safety",
        description="Auto-filter claims from LLM to meet accuracy and safety guarantees.",
    )
    parser.add_argument('-config_path', '-c', default='configs/default.toml', help="Config for construction.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    config = get_config(args.config_path)

    rng = np.random.default_rng(seed=0)

    # annotate dataset
    dataset = load_dataset(config)

    # split dataset into train / validation / test
    dataset_train, dataset_valid, dataset_test = split_dataset(
        dataset,
        train_perc=config.dataset.train_percent, 
        valid_perc=config.dataset.valid_percent
    )

    # identify features for scoring 
    score_features_v = [np.zeros((len(u['claims']), 1)) for u in dataset_valid]
    score_features_te = [np.zeros((len(u['claims']), 1)) for u in dataset_test]

    # fit scoring function using training set (or just take it from the model)
    score_fn = lambda x: rng.uniform(size=len(x)) # TODO: dumb one for now.

    # obtain min prob of each invalid claim in each 
    scores_valid = [score_fn(feats) for feats in score_features_v]
    conf_scores_valid = compute_conformity_scores(dataset_valid, scores_valid)

    scores_test = [score_fn(feats) for feats in score_features_te]

    # fit error probability function using training set (or just define it?)
    # we want to be more sure about correctness on more sensitive prompts
    alpha_fn = lambda x: [config.conformal.alpha] * len(x) # TODO: dumb one for now.

    # identify features for conditional calibration
    conf_features_v = np.zeros((len(dataset_valid),1)) 
    conf_features_te = np.zeros((len(dataset_test),1)) 

    # calibrate a threshold on the validation set
    thresholds = calibrate_thresholds(
        conf_features_te,
        conf_features_v,
        conf_scores_valid,
        alpha_fn
    )

    dataset_test = conformal_filter(
        dataset_test, 
        scores_test, 
        thresholds
    )

    print("Merging filtered responses.")
    dataset_test = merge_claims(
        dataset_test,
        config.model.parser.name
    )

    print(dataset_test[0]['response'])
    print(dataset_test[0]['filtered_response'])



    

