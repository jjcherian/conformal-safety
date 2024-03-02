from tqdm import tqdm
from typing import List, Tuple

import json
import numpy as np
import os

from query import query_llm
from llm_utils import parse_responses

def get_prompts(
    dataset : str
) -> List:
    if dataset.lower() == "factscore":
        with open('data/factscore_names.txt', 'r') as fp:
            names = fp.readlines()
        prompts = [
            f"Please write one biographical paragraph about {name}."
            for name in names
        ]
        return prompts
    else:
        raise ValueError("Unsupported data set.")
    

def load_dataset(
    config : dict
) -> List:
    cache_path = f"data/{config.dataset.name.lower()}_annotations.json"
    if os.path.isfile(cache_path):
        with open(cache_path, 'r') as fp:
            dataset = json.load(fp)
    else:
        response_path = f"data/{config.dataset.name.lower()}_responses.json"
        if os.path.isfile(response_path):
            with open(response_path, 'r') as fp:
                outputs = json.load(fp)
        else:
            prompts = get_prompts(config.dataset.name)
            outputs = []
            print("Fetching (possibly invalid) responses.")
            for p in tqdm(prompts):
                output = query_llm([p], config.model.responder.name, n_samples=1)
                outputs.append(output)
            outputs = [{'prompt': p, 'response': o[0]['message']} 
                         for p, o in zip(prompts, outputs)] # first output is the response we will filter
            with open(response_path, 'w') as fp:
                json.dump(outputs, fp)

        print("Parsing and annotating responses.")
        dataset = parse_responses(
            outputs,
            parser_config=config.model.parser.name,
            annotate=True,
            annotator_config=config.model.annotator.name
        )

        with open(cache_path, 'w') as fp:
            json.dump(dataset, fp)
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