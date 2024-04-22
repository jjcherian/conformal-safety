from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Tuple

import json
import numpy as np
import os

from atomizer import Atomizer
from gpt import GPTClient
from scorer import Scorer

def get_prompts(
    dataset : str
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
    if dataset.lower() == "factscore_v3":
        with open('data/factscore_v3_names.txt', 'r') as fp:
            names = fp.readlines()
        names = [name.strip() for name in names]
        prompts = [
            f"Please write one biographical paragraph about {name.strip()}."
            for name in names
        ]
        return names, prompts
    else:
        raise ValueError("Unsupported data set.")
    

def load_dataset(
    config : dict
) -> List:
    
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
    
    responder.cache_outputs(
        prompts,
        np.zeros((len(responses),), dtype=int),
        responses
    )

    responder.save_cache()

    responses = [r[0] for r in responses]


    outputs = [{'prompt': p, 'response': o['message']} 
                    for p, o in zip(prompts, responses)] # first output is the response we will filter

    atomizer_client = GPTClient(config.model.parser.cache_path, model=config.model.parser.name)

    atomizer = Atomizer(atomizer_client, demo_dir='data/demos')

    with ThreadPoolExecutor(max_workers=5) as executor:
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
    
    atomic_facts = [[fact for _, facts in atom[0] for fact in facts]for atom in atoms]
    
    
    # time to annotate responses using factscore code

    scorer_client = GPTClient(config.model.annotator.cache_path, model=config.model.annotator.name)
    scorer = Scorer(scorer_client, config, model_name="retrieval")
    scorer_inputs = [(topic, output['response'], fact) for topic, output, fact in zip(topics, outputs, atomic_facts)]
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