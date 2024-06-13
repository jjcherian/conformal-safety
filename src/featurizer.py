import numpy as np
import json

from functools import partial
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import client

from typing import Dict, List, Tuple


SELF_ASSESS_PROMPT = 'You will get a list of claims and the original prompt that motivated these claims. For each claim, assess the probability of correctness. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "gpt-score":[SCORE]}. Directly return the jsonl with NO explanation or ANY other formatting. For the [SCORE], return the esimated probability of correctness to three significant figures.\n'

SELF_BOOL_PROMPT = 'You will get a list of claims and the original prompt that motivated these claims. For each claim, assess the correctness. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "gpt-bool":[BOOL]}. Directly return the jsonl with NO explanation or ANY other formatting. For the [BOOL], return "T" or "F" in quotes so that it is valid json.\n'

MAX_WORKERS = 20

def get_features(
        dataset: List[Dict],
        config : Dict
) -> np.ndarray:
    from gpt import GPTClient
    feature_names = config.model.prob.features
    all_features = []
    if 'frequency' in feature_names:
        client = GPTClient(f'.cache/{config.dataset.name}_frequency.pkl')


        with ThreadPoolExecutor(max_workers=5) as executor:
            frequencies = list(
                tqdm(
                    executor.map(
                        lambda x: get_frequency(client, [af['atom'] for af in x['atomic_facts']], x['prompt'], config.model.prob.frequency.model),
                        dataset
                    ),
                    total=len(dataset)
                )
            )
        client.save_cache()
        all_features.append(np.concatenate(frequencies).reshape(-1,1))
    
    if 'selfeval' in feature_names:

        eval_client = GPTClient(f'.cache/{config.dataset.name}_self_evals.pkl')

        with ThreadPoolExecutor(max_workers=25) as executor:
            self_evals = list(
                tqdm(
                    executor.map(
                        lambda x: get_self_eval(x['prompt'], [af['atom'] for af in x['atomic_facts']], eval_client),
                        dataset
                    ),
                    total=len(dataset)
                )
            )
        eval_client.save_cache()
        all_features.append(np.concatenate(self_evals).reshape(-1,1))

    features = np.concatenate(
        all_features,
        axis=1
    )
    return features

# def get_features(
#         dataset : List[Dict], 
#         config : Dict
# ) -> np.ndarray:
#     feature_names = config.features
#     num_claims = np.sum([len(dat['claims']) for dat in dataset])
#     all_features = []
#     for feat in feature_names:
#         if feat == "embedding":
#             embeds = np.zeros((num_claims, int(config.embedding.n_dimensions)))
#             print("Fetching embeddings.")
#             embedding_func = partial(get_embedding, model=config.embedding.model, n_dim=config.embedding.n_dimensions)
#             res = process_map(embedding_func, [dat['claims'] for dat in dataset], max_workers=MAX_WORKERS)
#             i = 0
#             for dat in tqdm(dataset):
#                 len_dat = len(dat['claims'])
#                 embeds[i:(i + len_dat)] = get_embedding(dat['claims'], config.embedding.model, config.embedding.n_dimensions)
#                 i += len_dat
#             all_features.append(embeds)

#         elif feat == "selfeval":
#             print("Fetching selfevals.")
#             evals = np.zeros((num_claims, 1))
#             selfeval_func = partial(get_self_eval, model=config.selfeval.model.name)
#             res = process_map(selfeval_func, dataset, max_workers=MAX_WORKERS)
#             i = 0
#             for dat in tqdm(dataset):
#                 len_dat = len(dat['claims'])
#                 evals[i:(i + len_dat)] = get_self_eval(dat['claims'], dat['prompt'], config.selfeval.model.name)
#                 i += len_dat
#             all_features.append(evals)
#         elif feat == "frequency":
#             print("Fetching frequency.")
#             freqs = np.zeros(((num_claims), 1))
#             i = 0
#             for dat in tqdm(dataset):
#                 len_dat = len(dat['claims'])
#                 freqs[i:(i + len_dat)] = get_frequency(dat['claims'], dat['prompt'], config.frequency.model.n_samples, config.frequency.model.name)
#                 i += len_dat
#             all_features.append(freqs)
#         else:
#             raise ValueError(f"{feat} not supported.")
#     return np.concatenate(all_features, axis=1)


def get_embedding(
        subclaims : List[str], 
        client : client.Client, # needs to be embedding client not *GPT* client
        n_dim : int = 8
) -> np.ndarray:
    raise ValueError("not supported yet")
    embeddings = []
    for claim in subclaims:
        msg = claim['message'].replace('\n', ' ')
        embed = client.query(msg)
        embeddings.append(embed[:n_dim])
    return np.asarray(embeddings)


def _eval_self(
        prompt : str,
        subclaims : List,
        client : client.Client,
        err_msg : str = None
) -> Tuple[Tuple[str, List], np.ndarray]:
    claim_string = "\n".join(
        [str(i) + ": " + fact for i, fact in enumerate(subclaims)]
    )
    self_eval_prompt = SELF_ASSESS_PROMPT
    self_eval_prompt += f"The original prompt is: {prompt}.\n"
    self_eval_prompt += f"The claims are: {claim_string}.\n"

    if err_msg is not None:
        self_eval_prompt += "\n" + err_msg

    self_evals = client.query(self_eval_prompt)
    parsed_evals = self_evals[0]['message']
    parsed_evals = parsed_evals.replace("```jsonl\n", "")
    parsed_evals = parsed_evals.replace("```", "")
    final_evals = np.zeros((len(parsed_evals.splitlines()),))
    try:
        assert len(final_evals) == len(subclaims)
    except AssertionError:
        if err_msg is not None and 'exactly' in err_msg:
            print(f"I'm giving up on {claim_string} and {parsed_evals}, since I already retried this.")
            return (None, None), None
        err_msg = f"IMPORTANT: This is a retry. Make sure you return exactly {len(subclaims)} lines of JSON."
        print(err_msg)
        return _eval_self(prompt, subclaims, client, err_msg=err_msg)
    try:
        for line in parsed_evals.splitlines():
            eval = json.loads(line)
            idx = int(eval["id"])
            final_evals[idx] += float(eval["gpt-score"])
    except Exception as ex:
        if err_msg is not None and 'requested' in err_msg:
            print(f"I'm giving up on {claim_string} and {parsed_evals}, since I already retried this.")
            return (None, None), None
        err_msg = f"IMPORTANT: This is a retry. Make sure you return the lines in the requested JSON format with NO additional formatting."
        print(err_msg)
        return _eval_self(prompt, subclaims, client, err_msg=err_msg)
    return (self_eval_prompt, self_evals), final_evals


def get_self_eval(
        prompt : str, 
        subclaims : List[str], 
        client : client.Client
) -> np.ndarray:
    all_evals = _eval_self(
        prompt,
        subclaims,
        client
    )

    to_cache = all_evals[0]

    if to_cache[0] is None:
        return -1 * np.ones((len(subclaims),)) # -1 prob is error

    client.cache_outputs(
        [to_cache[0]],
        np.zeros((1,), dtype=int),
        [to_cache[1]]
    )

    return all_evals[1]

def _bool_self(
        prompt : str,
        subclaims : List,
        client : client.Client,
        err_msg : str = None
) -> Tuple[Tuple[str, List], np.ndarray]:
    claim_string = "\n".join(
        [str(i) + ": " + fact for i, fact in enumerate(subclaims)]
    )
    self_eval_prompt = SELF_BOOL_PROMPT
    self_eval_prompt += f"The original prompt is: {prompt}.\n"
    self_eval_prompt += f"The claims are: {claim_string}.\n"

    if err_msg is not None:
        self_eval_prompt += "\n" + err_msg

    self_evals = client.query(self_eval_prompt)
    parsed_evals = self_evals[0]['message']
    parsed_evals = parsed_evals.replace("```jsonl\n", "")
    parsed_evals = parsed_evals.replace("```", "")
    final_evals = ['T' for i in range(len(parsed_evals.splitlines()))]
    try:
        assert len(final_evals) == len(subclaims)
    except AssertionError:
        if err_msg is not None and 'exactly' in err_msg:
            print(f"I'm giving up on {claim_string} and {parsed_evals}, since I already retried this.")
            return (None, None), None
        err_msg = f"IMPORTANT: This is a retry. Make sure you return exactly {len(subclaims)} lines of JSON."
        print(err_msg)
        return _bool_self(prompt, subclaims, client, err_msg=err_msg)
    try:
        for line in parsed_evals.splitlines():
            eval = json.loads(line)
            idx = int(eval["id"])
            final_evals[idx] = eval["gpt-bool"]
    except Exception as ex:
        if err_msg is not None and 'requested' in err_msg:
            print(f"I'm giving up on {claim_string} and {parsed_evals}, since I already retried this.")
            return (None, None), None
        err_msg = f"IMPORTANT: This is a retry. Make sure you return the lines in the requested JSON format with NO additional formatting."
        print(err_msg)
        return _bool_self(prompt, subclaims, client, err_msg=err_msg)
    return (self_eval_prompt, self_evals), final_evals


def get_bool_eval(
        prompt : str, 
        subclaims : List[str], 
        client : client.Client
) -> np.ndarray:
    all_evals = _bool_self(
        prompt,
        subclaims,
        client
    )

    to_cache = all_evals[0]

    if to_cache[0] is None:
        return -1 * np.ones((len(subclaims),)) # -1 prob is error
    client.cache_outputs(
        [to_cache[0]],
        np.zeros((1,), dtype=int),
        [to_cache[1]]
    )

    return all_evals[1]


def _eval_support(
        output : str, 
        subclaims : List, 
        client : client.Client, 
        err_msg : str = None
) -> Tuple[Tuple[str, List], np.ndarray]:
    claim_string = "\n".join(
        [str(i) + ": " + fact for i, fact in enumerate(subclaims)]
    )
    counting_prompt = (
        'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "score":[SCORE]}. Directly return the jsonl with NO explanation or ANY other formatting. For the [SCORE], return 1 for supports, -1 for contradicts, and 0 for unrelated. The claims are:\n'
        + claim_string
        + "\n\nThe text is:\n"
        + output
    )
    if err_msg is not None:
        counting_prompt += "\n" + err_msg

    support_scores = client.query(counting_prompt)
    parsed_scores = support_scores[0]['message']
    parsed_scores = parsed_scores.replace("```jsonl\n", "")
    parsed_scores = parsed_scores.replace("```", "")
    final_scores = np.zeros((len(parsed_scores.splitlines()),))
    try:
        assert len(final_scores) == len(subclaims)
    except AssertionError:
        if err_msg is not None and 'exactly' in err_msg:
            print(f"I'm giving up on {claim_string} and {parsed_scores}, since I already retried this.")
            return (None, None), None
        err_msg = f"IMPORTANT: This is a retry. Make sure you return exactly {len(subclaims)} lines of JSON."
        print(err_msg)
        return _eval_support(output, subclaims, client, err_msg=err_msg)
    try:
        for line in parsed_scores.splitlines():
            score = json.loads(line)
            idx = int(score["id"])
            final_scores[idx] += float(score["score"])
    except Exception as ex:
        if err_msg is not None and 'requested' in err_msg:
            print(f"I'm giving up on {claim_string} and {parsed_scores}, since I already retried this.")
            return (None, None), None
        err_msg = f"IMPORTANT: This is a retry. Make sure you return the lines in the requested JSON format with NO additional formatting."
        print(err_msg)
        return _eval_support(output, subclaims, client, err_msg=err_msg)
    return (counting_prompt, support_scores), final_scores
    

def get_frequency(
        client : client.Client, 
        subclaims : List, 
        prompt : str, 
        config : dict
) -> np.ndarray:
    """
    Returns a vector of (frequency) scores corresponding to each entry of the subclaims list.
    """
    # Generate n_samples alternate outputs with temperature 1.0.
    alternate_outputs = client.query(
        prompt, 1, n_samples=config.n_samples, temperature=config.temperature
    )
    client.cache_outputs(
        [prompt],
        [int(1)],
        [alternate_outputs]
    )

    alternate_outputs = [o['message'] for o in alternate_outputs]

    with ThreadPoolExecutor(max_workers=config.n_samples) as executor:
        all_scores = list(
            executor.map(
                lambda x : _eval_support(x, subclaims, client),
                alternate_outputs
            )
        )

    # to_cache = [s[0] for s in all_scores if s[0][0] is not None]

    # client.cache_outputs(
    #     [c[0] for c in to_cache],
    #     np.zeros((len(to_cache),), dtype=int),
    #     [c[1] for c in to_cache]
    # )

    # TODO: error handling if this is all empty?
    parsed_scores = np.mean([s[1] for s in all_scores if s[1] is not None], axis=0)

    return parsed_scores