import argparse
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from config import get_config

from featurizer import get_frequency, get_self_eval
from gpt import GPTClient

from atomizer import text_to_sentences
from dataset import get_prompts
from scorer import Scorer

import ray

def parse_args():
    parser = argparse.ArgumentParser(
        prog="conformal-safety",
        description="Auto-filter claims from LLM to meet accuracy and safety guarantees.",
    )
    parser.add_argument('-config_path', '-c', default='configs/default.toml', help="Config for construction.")
    args = parser.parse_args()
    return args

def find_unique_element(lst, condition, approx_index):
    # Check the approximate index first
    if condition(lst[approx_index]):
        return approx_index
    
    # Initialize left and right pointers
    left = approx_index - 1
    right = approx_index + 1
    
    # Expand outwards from the approximate index
    while left >= 0 or right < len(lst):
        if left >= 0 and condition(lst[left]):
            return left
        if right < len(lst) and condition(lst[right]):
            return right
        left -= 1
        right += 1
    
    # If no element satisfies the condition, return None or raise an exception
    return None


@ray.remote
def parallel_scorer(*args, **kwargs):
    return None
    return run_experiment(*args, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.config_path)

    import IPython; IPython.embed()
    responder = GPTClient(config.model.responder.cache_path)
        
    topics, prompts = get_prompts(config.dataset.name)

    with ThreadPoolExecutor(max_workers=25) as executor:
        responses = list(
            tqdm(
                executor.map(
                    lambda x : responder.query(x),
                    prompts
                ),
                total=len(prompts)
            )
        )
    
    responses = [r[0] for r in responses]


    outputs = [{'prompt': p, 'response': o['message']} 
                    for p, o in zip(prompts, responses)] # first output is the response we will filter
    
    print("Loading atomizer.")
    atomizer_client = GPTClient(config.model.parser.cache_path, model=config.model.parser.name)
    
    responder_cache = responder.cache_dict
    messages = []
    for val in responder_cache.values():
        messages.append(val[0]['message'])

    atomizer_cache = atomizer_client.cache_dict
    idx_guess = 0
    atomic_facts = [[] for _ in range(len(messages))]
    for k in tqdm(atomizer_cache.keys()):
        atomized_msg = atomizer_cache[k][0]['message']
        atomized_facts = text_to_sentences(atomized_msg)
        sentence = k.split('\n')[-1].split('facts:')[-1].strip()[:-2]
        cur_idx = find_unique_element(messages, lambda x: sentence in x, approx_index=idx_guess)
        if cur_idx is None: # TODO: TERRIBLE SPECIAL CASING that I looked at by hand...
            if idx_guess == 4151:
                cur_idx = 4152
            else:
                cur_idx = idx_guess
        idx_guess = cur_idx
        atomic_facts[cur_idx].extend(atomized_facts)

    # time to annotate responses using factscore code
    print("Loading annotator.")
    scorer_client = GPTClient(config.model.annotator.cache_path, model=config.model.annotator.name)
    scorer = Scorer(scorer_client, config, model_name="retrieval")

    scorer_inputs = [(topic, output['response'], fact) for topic, output, fact in zip(topics, outputs, atomic_facts)]

    import IPython; IPython.embed()


    # connect to cluster
    ray.init(address="auto")

    results = []

    for seed in range(args.seed, args.seed + args.n_trials):
        if args.type == 'coverage':
            result = parallel_coverage_experiment.remote(
                (X, Y), n_test, n_calib, alpha, methods=args.methods, seed=seed
            )
        else:
            result = parallel_experiment.remote(
                (X, Y), n_test, n_calib, alpha, methods=args.methods, seed=seed
            )
        results.append(result)
    
    trial_results = ray.get(results)

    with ThreadPoolExecutor(max_workers=1) as executor:
        scores = list(
            tqdm(
                executor.map(
                    lambda x : scorer.get_score(*x),
                    scorer_inputs
                ),
                total=len(scorer_inputs)
            )
        )
    scorer.save_cache()

    dataset = []

    for p, r, s in zip(prompts, responses, scores):
        data_pt = {
            'prompt': p,
            'response': r,
            'atomic_facts': s['decisions'][0]
        }
        dataset.append(data_pt)

    import IPython
    IPython.embed()

    # client = GPTClient(f'.cache/{config.dataset.name}_frequency.pkl')

    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     frequencies = list(
    #         tqdm(
    #             executor.map(
    #                 lambda x: get_frequency(client, [af['atom'] for af in x['atomic_facts']], x['prompt'], config.model.prob.frequency.model),
    #                 dataset
    #             ),
    #             total=len(dataset)
    #         )
    #     )
    # client.save_cache()

    # eval_client = GPTClient(f'.cache/{config.dataset.name}_self_evals.pkl')

    # with ThreadPoolExecutor(max_workers=25) as executor:
    #     self_evals = list(
    #         tqdm(
    #             executor.map(
    #                 lambda x: get_self_eval(x['prompt'], [af['atom'] for af in x['atomic_facts']], eval_client),
    #                 dataset
    #             ),
    #             total=len(dataset)
    #         )
    #     )
    # eval_client.save_cache()

    # features = np.concatenate(
    #     [
    #         np.concatenate(frequencies).reshape(-1,1),
    #         np.concatenate(self_evals).reshape(-1,1)
    #     ],
    #     axis=1
    # )