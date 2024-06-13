from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Tuple

import json
import pandas as pd
import numpy as np
import os

from atomizer import Atomizer, text_to_sentences
from gpt import GPTClient
from scorer import Scorer

def get_prompts(
    dataset : str,
    data_path : str = None
) -> List:
    if dataset.lower() == "factscore":
        with open('data/factscore_names.txt', 'r') as fp:
            names = fp.readlines()
        names = [name.strip() for name in names]
        prompts = [
            f"Please write one biographical paragraph about {name.strip()}."
            for name in names
        ]
        return names, prompts
    if dataset.lower() == "factscore_v2":
        with open('data/factscore_v2_names.txt', 'r') as fp:
            names = fp.readlines()
        names = [name.strip() for name in names]
        prompts = [
            f"Please write one biographical paragraph about {name.strip()}."
            for name in names
        ]
        return names, prompts
    if dataset.lower() == "factscore_v3":
        with open('data/factscore_v3_names.txt', 'r') as fp:
            names = fp.readlines()
        names = [name.strip() for name in names]
        prompts = [
            f"Please write one biographical paragraph about {name.strip()}."
            for name in names
        ]
        return names, prompts
    
    if dataset.lower() == "factscore_final":
        df = pd.read_csv(data_path, index_col=0)
        names = set([n.strip() for n in df['Name']])
        prompts = [
            f"Please write one biographical paragraph about {name.strip()}."
            for name in names
        ]
        return names, prompts

    if dataset.lower() == "medlfqa":
        datasets = {}

        suffix = "_test_MedLFQA.jsonl"

        dataset_dir = "/Users/cherian/Projects/OLAPH/MedLFQA"
        for path in os.listdir(dataset_dir):
            if "MedLFQA" not in path:
                continue
            dataset_name = path[:-len(suffix)]
            with open(os.path.join(dataset_dir, path), 'r') as fp:
                datasets[dataset_name] = [json.loads(line) for line in fp.readlines()]

        prompts = []
        for _, dataset in datasets.items():
            prompts += [pt['Question'] for pt in dataset]
        prompts = list(set(prompts))
        return prompts, prompts
    
    if dataset.lower() == "medlfqav2":
        datasets = {}

        suffix = ".jsonl"

        for filename in os.listdir(data_path):
            dataset_name = filename[:-len(suffix)]
            with open(os.path.join(data_path, filename), 'r') as fp:
                datasets[dataset_name] = [json.loads(line) for line in fp.readlines()]

        prompts = []
        for _, dataset in datasets.items():
            prompts += [pt['Question'] for pt in dataset]

        return prompts, prompts

    else:
        raise ValueError("Unsupported data set.")
    
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

def load_dataset(
    config : dict
) -> List:
    
    print("Loading responder.")
    responder = GPTClient(config.model.responder.cache_path)
        
    topics, prompts = get_prompts(config.dataset.name, config.dataset.path)

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
    
    # TODO: Uncomment me if I want to run fresh dataset...

    responder.cache_outputs(
        prompts,
        np.zeros((len(responses),), dtype=int),
        responses
    )

    responder.save_cache()

    responses = [r[0] for r in responses]


    outputs = [{'prompt': p, 'response': o['message']} 
                    for p, o in zip(prompts, responses)] # first output is the response we will filter

    import IPython; IPython.embed()
    print("Loading atomizer.")
    atomizer_client = GPTClient(config.model.parser.cache_path, model=config.model.parser.name)

    atomizer = Atomizer(atomizer_client, demo_dir='data/demos')
    
    CACHE_EXISTS = True

    if CACHE_EXISTS: # TODO: dumb hard-coded variable to side step the slow retrieval
        ordered_messages = [r['message'] for r in responses]

        responder_cache = responder.cache_dict
        messages = []
        for val in responder_cache.values():
            messages.append(val[0]['message'])

        atomizer_cache = atomizer_client.cache_dict
        idx_guess = 0
        atomic_facts = [[] for _ in range(len(messages))]
        atomic_facts_ph = [[] for _ in range(len(messages))]

        sentences = defaultdict(int)
        for k in tqdm(atomizer_cache.keys()):
            atomized_msg = atomizer_cache[k][0]['message']
            atomized_facts = text_to_sentences(atomized_msg)
            sentence = k.split('\n')[-1].split('facts:')[-1].strip()[:-2]
            cur_idx = -1
            sentences[sentence] += 1
            # if the sentence has appeared more than once we need to find the appropriate match...
            for i in range(sentences[sentence]):
                cur_idx = find_unique_element(messages[cur_idx + 1:], lambda x: sentence in x, approx_index=idx_guess)
            if cur_idx is None: # TODO: TERRIBLE SPECIAL CASING that I looked at by hand...
                raise ValueError()
                if idx_guess in (4148, 4149, 4150):
                    cur_idx = 4149
                elif cur_idx == 993:
                    cur_idx = 993
                else:
                    continue
            idx_guess = cur_idx
            atomic_facts[cur_idx].extend(atomized_facts)

        for af, msg in zip(atomic_facts, messages):
            if len(af) == 0:
                continue
            new_idx = ordered_messages.index(msg)
            atomic_facts_ph[new_idx] = af
        atomic_facts = atomic_facts_ph
        
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            atoms = list(
                tqdm(
                    executor.map(
                        lambda x : atomizer.run(*x),
                        [(o['response'],) for o in outputs]
                    ),
                    total=len(outputs)
                )
            )

        atomizer.save_cache()
        atomic_facts = [[fact for _, facts in atom[0] for fact in facts] for atom in atoms]
    
    
    dataset = []

    for p, r, af in zip(prompts, responses, atomic_facts):
        atoms = [{'atom': fact} for fact in af]
        data_pt = {'prompt': p, 'response': r, 'atomic_facts': atoms}
        dataset.append(data_pt)

    # time to annotate responses using factscore code
    print("Loading annotator.")
    scorer_client = GPTClient(config.model.annotator.cache_path, model=config.model.annotator.name)
    scorer = Scorer(scorer_client, config, model_name="retrieval")

    scorer_inputs = [(topic, output['response'], fact) for topic, output, fact in zip(topics, outputs, atomic_facts)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(
            tqdm(
                executor.map(
                    lambda x : scorer.get_score(*x, knowledge_source='medlfqa'),
                    scorer_inputs
                ),
                total=len(scorer_inputs)
            )
        )
    # scorer.save_cache()

    dataset = []

    for p, r, s in zip(prompts, responses, scores):
        data_pt = {
            'prompt': p,
            'response': r,
            'atomic_facts': s['decisions'][0]
        }
        dataset.append(data_pt)

    import IPython; IPython.embed()

    return dataset

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
    
    # Calculate lengths of each part based on percentages
    len1 = int(total_length * train_perc)
    len2 = int(total_length * valid_perc)
    
    # if rng passed in, shuffle the dataset
    if rng is not None:
        rng.shuffle(dataset)

    # Split the list using slicing
    train_data = dataset[:len1]
    valid_data = dataset[len1:len1+len2]
    test_data = dataset[len1+len2:]
    
    return train_data, valid_data, test_data