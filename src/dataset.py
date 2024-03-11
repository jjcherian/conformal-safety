from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Tuple

import json
import numpy as np
import os

from atomizer import Atomizer
from gpt import GPTClient
from query import query_llm
from llm_utils import parse_responses

def get_prompts(
    dataset : str
) -> List:
    if dataset.lower() == "factscore":
        with open('data/factscore_names.txt', 'r') as fp:
            names = fp.readlines()
        prompts = [
            f"Please write one biographical paragraph about {name.strip()}."
            for name in names
        ]
        return prompts
    else:
        raise ValueError("Unsupported data set.")
    

def load_dataset(
    config : dict
) -> List:
    
    responder = GPTClient(f'.cache/{config.dataset.name}_responses.pkl')

    prompts = get_prompts(config.dataset.name)

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
                    lambda x : atomizer.run(x),
                    [o['response'] for o in outputs]
                ),
                total=len(outputs)
            )
        )

    atomizer.client.save_cache()
    
    import IPython; IPython.embed()

    print("Parsing and annotating responses. Should change this to just subprocess factscorer.py because they did it much better than me.")
    # dataset = parse_responses(
    #     outputs,
    #     parser_config=config.model.parser.name,
    #     annotate=True,
    #     annotator_config=config.model.annotator.name
    # )

    # with open(cache_path, 'w') as fp:
    #     json.dump(dataset, fp)
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