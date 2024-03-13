import argparse
import numpy as np

from config import get_config
from conformal import compute_conformity_scores, calibrate_thresholds, conformal_filter, assess_factscore_coverage
from dataset import load_dataset, split_dataset
from featurizer import get_features
from llm_utils import merge_claims
from prob_model import fit_model
from gpt import GPTClient


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

    rng = np.random.default_rng(seed=config.dataset.seed)

    # annotate dataset
    dataset = load_dataset(config)

    # split dataset into train / validation / test
    dataset_train, dataset_valid, dataset_test = split_dataset(
        dataset,
        train_perc=config.dataset.train_percent, 
        valid_perc=config.dataset.valid_percent,
        rng=rng if config.dataset.randomize else None
    )

    X_train = get_features(dataset_train, config)

    y_train = np.concatenate([[c['is_supported'] for c in dat['atomic_facts']] for dat in dataset_train])
    y_train[y_train == True] = 1
    y_train[y_train == False] = 0
    y_train = y_train.astype(np.int8)

    X_valid = get_features(dataset_valid, config)
    y_valid = np.concatenate([[c['is_supported'] for c in dat['atomic_facts']] for dat in dataset_valid])
    y_valid[y_valid == True] = 1
    y_valid[y_valid == False] = 0
    y_valid = y_valid.astype(np.int8)
    splits_valid = np.cumsum([len(dat['atomic_facts']) for dat in dataset_valid])[:-1]

    X_test = get_features(dataset_test, config)
    y_test = np.concatenate([[c['is_supported'] for c in dat['atomic_facts']] for dat in dataset_test])
    y_test[y_test == True] = 1
    y_test[y_test == False] = 0
    y_test = y_test.astype(np.int8)
    splits_test = np.cumsum([len(dat['atomic_facts']) for dat in dataset_test])[:-1]

    model = fit_model(X_train, y_train, config, dataset_train, 
                      eval_dict={'X_valid': X_valid, 'X_test': X_test, 'dataset_valid': dataset_valid, 'splits_valid': splits_valid, 'splits_test': splits_test})

    scores_valid = model.predict_proba(X_valid)[:,1]
    scores_valid = np.array_split(scores_valid, splits_valid)

    scores_test = model.predict_proba(X_test)[:,1]
    scores_test = np.array_split(scores_test, splits_test)
    # identify features for scoring 
    score_features_v = [np.zeros((len(u['atomic_facts']), 1)) for u in dataset_valid]
    score_features_te = [np.zeros((len(u['atomic_facts']), 1)) for u in dataset_test]

    conf_scores_valid = compute_conformity_scores(dataset_valid, scores_valid)

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

    if config.dataset.name.lower() == "factscore":
        assess_factscore_coverage(dataset_test, config.conformal.alpha)
        
    print("Merging filtered responses.")

    merge_client = GPTClient(cache_file = config.model.merger.cache_path)
    merged_responses = merge_claims(
        dataset_test,
        merge_client
    )
    merge_client.save_cache()

    rand_idx = rng.integers(0, len(dataset_test))
    print(dataset_test[rand_idx]['response']['message'] + "\n")
    print(merged_responses[rand_idx])

    import IPython; IPython.embed()



    

