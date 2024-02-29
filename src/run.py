import argparse
from tqdm import tqdm
import numpy as np

from dataset import get_prompts
from query import query_llm


def parse_args():
    parser = argparse.ArgumentParser(
        prog="conformal-safety",
        description="Auto-filter claims from LLM to meet accuracy and safety guarantees.",
    )
    ## should change this to passing in a json config?? seems more generalizable especially if we want to run lots of experiments
    ## batched onto a cluster?
    parser.add_argument('dataset', help="Dataset used.")
    parser.add_argument('-m', '--output_model', default="gpt-3.5-turbo")
    parser.add_argument('-p', '--parser_model', default="gpt-4-0125-preview")
    parser.add_argument('-a', '--annotator_model', default="gpt-4-0125-preview")

    args = parser.parse_args()
    return args


# TODO; these functions doesn't belong here...but where should I put it. TBD.
from typing import Callable, Dict, List, Tuple

def parse_responses(
    prompts : List[str],
    responses : List[str],
    parser_config : str,
    annotate : bool = False,
    annotator_config : str = None
):
    parsed_subclaims = []
    for i in tqdm(range(len(prompts))):
        prompt, response = prompts[i], responses[i]
        subclaims = get_subclaims(prompt, response, parser_config)
        if annotate:
            subclaims = add_annotations(prompt, subclaims, annotator_config)
        parsed_subclaims.append(subclaims)
    return parsed_subclaims

def get_subclaims(
        prompt : str, 
        response : str, 
        parser_config : str
) -> List[Dict]:
    from query import generate_subclaim_prompt, query_llm
    subclaim_prompt = generate_subclaim_prompt(prompt, response)
    subclaims = query_llm([subclaim_prompt], parser_config)[0] # get the first output
    subclaims = [{'message': c} for c in subclaims['message'].splitlines()]
    return subclaims

def add_annotations(
        prompt : str,
        subclaims : List[Dict],
        annotator_config : str
) -> List[Dict]:
    from query import generate_annotation_prompt, query_llm
    annotation_prompt = generate_annotation_prompt(prompt, subclaims)
    annotations = query_llm([annotation_prompt], annotator_config)[0]
    for a, subclaim in zip(annotations['message'].splitlines(), subclaims):
        subclaim['annotation'] = a
    return subclaims

def split_dataset(
    dataset : List,
    train_perc : float = 0.33,
    valid_perc : float = 0.33,
    rng : np.random.Generator = None
) -> Tuple[List, List, List]:
    """
    Splits dataset into three parts. Split into training and validation is specified here.
    """
    total_length = len(dataset)
    indices = np.arange(total_length)
    
    # Calculate lengths of each part based on percentages
    len1 = int(total_length * train_perc)
    len2 = int(total_length * valid_perc)
    
    # if rng passed in, shuffle the dataset
    if rng is not None:
        rng.shuffle(indices)

    # Split the list using slicing
    train_data = indices[:len1]
    valid_data = indices[len1:len1+len2]
    test_data = indices[len1+len2:]
    
    return train_data, valid_data, test_data

def compute_conformity_scores(
    claims_list : List,
    scores_list : List,
):
    annotations_list = [
        np.asarray([True if c['annotation'] in ('T', 'S') else False for c in claims])
        for claims in claims_list
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
    
def filter_claims(
    claims_test : List,
    scores_test : List,
    thresholds : List
) -> List:
    filtered_claims = [
        [c for c, s in zip(claims, scores) if s >= t] 
        for claims, scores, t in zip(claims_test, scores_test, thresholds)
    ]
    return filtered_claims

def merge_claims(
    prompts_list : List,
    claims_list : List,
    parser_config : str
) -> List:
    from query import generate_merge_prompt, query_llm
    merged_claims = []
    for i in tqdm(range(len(prompts_list))):
        prompt, claims = prompts_list[i], claims_list[i]
        merge_prompt = generate_merge_prompt(prompt, claims)
        output = query_llm([merge_prompt], parser_config)[0]
        merged_claims.append(output['message'])
    return merged_claims

if __name__ == "__main__":
    args = parse_args()

    # make these configurable
    TRAIN_PERC = 0.0
    VALID_PERC = 0.5
    ALPHA = 0.5

    rng = np.random.default_rng(seed=0)

    # annotate dataset

    prompts = get_prompts(args.dataset)[0:10]

    outputs = []
    print("Fetching (possibly invalid) responses.")
    for p in tqdm(prompts):
        output = query_llm([p], args.output_model, n_samples=1)
        outputs.append(output)
    responses = [o[0]['message'] for o in outputs] # first output is the response we will filter

    print("Parsing and annotating responses.")
    annotated_subclaims = parse_responses(
        prompts,
        responses,
        parser_config=args.parser_model,
        annotate=True,
        annotator_config=args.annotator_model
    )

    # split annotated outputs into train / validation / test
    indices_train, indices_valid, indices_test = split_dataset(
        annotated_subclaims,
        train_perc=TRAIN_PERC, 
        valid_perc=VALID_PERC
    )

    claims_train = [annotated_subclaims[i] for i in indices_train]
    prompts_train = [prompts[i] for i in indices_train]

    claims_valid = [annotated_subclaims[i] for i in indices_valid]
    prompts_valid = [prompts[i] for i in indices_valid]

    claims_test = [annotated_subclaims[i] for i in indices_test]
    prompts_test = [prompts[i] for i in indices_test]

    # identify features for scoring 
    score_features_v = [np.zeros((len(c), 1)) for c in claims_valid]
    score_features_te = [np.zeros((len(c), 1)) for c in claims_test]

    # fit scoring function using training set (or just take it from the model)
    score_fn = lambda x: rng.uniform(size=len(x)) # TODO: dumb one for now.

    # obtain min prob of each invalid claim in each 
    scores_valid = [score_fn(feats) for feats in score_features_v]
    conf_scores_valid = compute_conformity_scores(claims_valid, scores_valid)

    scores_test = [score_fn(feats) for feats in score_features_te]

    # fit error probability function using training set (or just define it?)
    # we want to be more sure about correctness on more sensitive prompts
    alpha_fn = lambda x: [ALPHA] * len(x) # TODO: dumb one for now.

    # identify features for conditional calibration
    conf_features_v = np.zeros((len(claims_valid),1)) 
    conf_features_te = np.zeros((len(claims_test),1)) 

    # calibrate a threshold on the validation set
    thresholds = calibrate_thresholds(
        conf_features_te,
        conf_features_v,
        conf_scores_valid,
        alpha_fn
    )

    filtered_claims_test = filter_claims(
        claims_test, 
        scores_test, 
        thresholds
    )

    print("Merging filtered responses.")
    merged_claims = merge_claims(
        prompts_test,
        filtered_claims_test,
        args.parser_model
    )

    print(merged_claims)



    

