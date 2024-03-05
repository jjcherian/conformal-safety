import numpy as np
import json

from tqdm import tqdm

from query import query_llm

SELF_ASSESS_PROMPT = "Given the original prompt, please evaluate the following set of small, independent claims, and return the output as a jsonl, where each line is {subclaim: CLAIM, gpt-score: CONF}.\n CLAIM should be a raw string (not a list containing a string) corresponding to ONE subclaim. The confidence score CONF should represent your estimated probabilty of correctness of the claim to three significant figures. The original prompt is: "

def get_features(dataset, config):
    feature_names = config.features
    num_claims = np.sum([len(dat['claims']) for dat in dataset])
    all_features = []
    for feat in feature_names:
        if feat == "embedding":
            embeds = np.zeros((num_claims, int(config.embedding.n_dimensions)))
            print("Fetching embeddings.")
            i = 0
            for dat in tqdm(dataset):
                len_dat = len(dat['claims'])
                embeds[i:(i + len_dat)] = get_embedding(dat['claims'], config.embedding.model, config.embedding.n_dimensions)
                i += len_dat
            all_features.append(embeds)

        elif feat == "selfeval":
            print("Fetching selfevals.")
            evals = np.zeros((num_claims, 1))
            i = 0
            for dat in tqdm(dataset):
                len_dat = len(dat['claims'])
                evals[i:(i + len_dat)] = get_self_eval(dat['claims'], dat['prompt'], config.selfeval.model.name)
                i += len_dat
            all_features.append(evals)
        elif feat == "frequency":
            print("Fetching frequency.")
            freqs = np.zeros(((num_claims), 1))
            i = 0
            for dat in tqdm(dataset):
                len_dat = len(dat['claims'])
                freqs[i:(i + len_dat)] = get_frequency(dat['claims'], dat['prompt'], config.frequency.model.n_samples, config.frequency.model.name)
                i += len_dat
            all_features.append(freqs)
        else:
            raise ValueError(f"{feat} not supported.")
    return np.concatenate(all_features, axis=1)


def get_embedding(subclaims, model, n_dim=8):
    embeddings = []
    for claim in subclaims:
        msg = claim['message'].replace('\n', ' ')
        embed = query_llm([msg], model)
        embeddings.append(embed[:n_dim])
    return np.asarray(embeddings)

def get_self_eval(subclaims, prompt, model):
    scores = []
    self_eval_prompt = SELF_ASSESS_PROMPT + prompt + "\n"
    for i, claim in enumerate(subclaims):
        msg = claim['message']
        self_eval_prompt = self_eval_prompt + f"Claim {i}: {msg}\n"
    output = query_llm([self_eval_prompt], model)[0]['message']
    for line in output.splitlines():
        scores.append(json.loads(line)["gpt-score"])
    return np.asarray(scores).reshape(-1,1)


def get_frequency(subclaims, prompt, n_samples, model):
    """
    Returns a vector of (frequency) scores corresponding to each entry of the subclaims list.
    """
    # Generate n_samples alternate outputs with temperature 1.0.
    alternate_outputs = query_llm(
        prompts=[prompt],
        model=model,
        n_samples=n_samples
    )
    alternate_outputs = [choice['message'] for choice in alternate_outputs]
    claim_string = "\n".join(
        [str(i) + ": " + fact["message"] for i, fact in enumerate(subclaims)]
    )

    # Count the number of times the alternate outputs support the sub-claims (using LM).
    # TODO: should this really be -1, 0, 1? Before it was 0, 1.
    final_scores = [0.0] * len(subclaims)
    for output in alternate_outputs:
        counting_prompt = (
            'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "score":[SCORE]}. Directly return the jsonl with no explanation or other formatting. For the [SCORE], return 1 for supports, -1 for contradicts, and 0 for unrelated. The claims are:\n'
            + claim_string
            + "\n\nThe text is:\n"
            + output
        )
        output = query_llm(
            [counting_prompt], model, max_tokens=1000, temperature=0
        )[0]['message']
        output = output.replace("```jsonl\n", "")
        output = output.replace("```", "")
        try:
            for i, line in enumerate(output.splitlines()):
                scores = json.loads(line)
                idx = int(scores["id"])
                final_scores[idx] += float(scores["score"])
        except Exception as ex:
            print(ex)
            print("Failed to parse as jsonl")
            print(output)

    return np.asarray(final_scores).reshape(-1,1)