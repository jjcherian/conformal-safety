import os

from openai import OpenAI
from query import query_model, get_subclaims, get_truth

if __name__ == "__main__":
    client = OpenAI()

    prompt = "Write one paragraph about Emmanuel Candes. Please ensure that at least one sub-claim is false."
    model = "gpt-3.5-turbo"
    output = query_model(client, prompt, model)

    subclaims = get_subclaims(client, prompt, output, model)

    truth_scores = get_truth(client, "gpt-4", prompt, subclaims)
    
    for subclaim, ts in zip(subclaims, truth_scores):
        subclaim['truth'] = ts['score']
    print(subclaims)