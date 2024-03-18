import json
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List

from query import (
    generate_subclaim_prompt, generate_annotation_prompt, 
    generate_merge_prompt, query_llm
)

import client

MERGE_PROMPT = "You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and use ALL of the facts provided. If no facts are given, reply and say that you don't know enough to respond.\n"


def parse_responses(
    outputs : List[Dict],
    parser_config : str,
    annotate : bool = False,
    annotator_config : str = None
):
    for output in tqdm(outputs):
        prompt, response = output["prompt"], output["response"]
        subclaims = get_subclaims(prompt, response, parser_config)
        if annotate:
            subclaims = add_annotations(prompt, subclaims, annotator_config)
        output["claims"] = subclaims
    return outputs

def get_subclaims(
        prompt : str, 
        response : str, 
        parser_config : str
) -> List[Dict]:
    subclaim_prompt = generate_subclaim_prompt(prompt, response)
    subclaims = query_llm([subclaim_prompt], parser_config)[0] # get the first output
    subclaims = [{'message': c} for c in subclaims['message'].splitlines()]
    return subclaims

def add_annotations(
        prompt : str,
        subclaims : List[Dict],
        annotator_config : str
) -> List[Dict]:
    annotation_prompt = generate_annotation_prompt(prompt, subclaims)
    annotations = query_llm([annotation_prompt], annotator_config)[0]
    annotations = annotations['message'].splitlines()
    num_retries = 0
    while len(annotations) != len(subclaims):
        print(f"Annotation length does not match subclaims for {prompt}. Retrying query.")
        annotations = query_llm([annotation_prompt], annotator_config)[0]
        annotations = annotations['message'].splitlines() 
        num_retries += 1
        if num_retries > 5:
            print("Giving up and assigning False to all subclaims.")
            annotations = ['F' for _ in subclaims]
    for a, subclaim in zip(annotations, subclaims):
        try:
            subclaim['annotation'] = json.loads(a)['value']
        except:
            import IPython; IPython.embed()
    return subclaims


def _concat_claims(
    subclaims : List[str]
) -> str:
    return "\n".join(
        f"{i}: {subclaim}" for i, subclaim in enumerate(subclaims) 
    )

def _get_merged_output(
    prompt : str,
    subclaims : List[str],
    client : client.Client
) -> str:
    final_prompt = MERGE_PROMPT + f"The original instruction was: {prompt}\n"

    final_prompt += f"The facts are: {_concat_claims(subclaims)}"

    output = client.query(final_prompt)

    return (final_prompt, output), output[0]['message']


def merge_claims(
    dataset : List,
    client : client.Client
) -> List:
    with ThreadPoolExecutor(max_workers=25) as executor:
        responses = list(
            tqdm(
                executor.map(
                    lambda x : _get_merged_output(x['prompt'], x['filtered_claims'], client),
                    dataset
                ),
                total=len(dataset)
            )
        )

    to_cache = [r[0] for r in responses]

    client.cache_outputs(
        [c[0] for c in to_cache],
        np.zeros((len(to_cache),), dtype=int),
        [c[1] for c in to_cache]
    )

    return [r[1] for r in responses]