import json
from tqdm import tqdm
from typing import Dict, List

from query import (
    generate_subclaim_prompt, generate_annotation_prompt, 
    generate_merge_prompt, query_llm
)

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


def merge_claims(
    dataset : List,
    parser_config : str
) -> List:
    for unit in tqdm(dataset):
        prompt, claims = unit['prompt'], unit['filtered_claims']
        merge_prompt = generate_merge_prompt(prompt, claims)
        output = query_llm([merge_prompt], parser_config)[0]
        unit['filtered_response'] = output['message']
    return dataset