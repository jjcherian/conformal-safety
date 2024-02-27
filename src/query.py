# this code is sayless.py in Mohri and Hashimoto, 2024.

import json

# Default prompt to break into subclaims.
BREAKDOWN_PROMPT = "Please breakdown the following input into a set of small, independent claims, and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n The confidence score [CONF] should represent your estimated probabilty of correctness of the claim to three significant figures. The input is: "

ORIGINAL_PROMPT = "The following was the original instruction given to the answerer. The original prompt was: "

VALIDITY_PROMPT = "You will get an instruction and a set of claims made in response to that instruction. Determine whether each claim is true or false. Directly return a jsonl with no additional explanation or formatting. Each line is {id:[CLAIM_ID], score:[SCORE]}. For the [SCORE], return 1 for True and 0 for False. "

def query_model(client, prompt, model, max_tokens=1000, temperature=0, n_samples=1):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n_samples,
    )
    return completion.choices[0].message.content


def say_less(client, prompt, model, output, threshold):
    """
    say_less takes in the model output y, breaks it down into subclaims, and removes sub-claims up to the threshold value.
    The subclaims are scored by counting (using an LM) how many times they appear from 5 other sampled outputs. This is done
    in get_frequency_scores.
    """
    subclaims = get_subclaims(client, output, model)

    frequency_scores = get_frequency_scores(client, subclaims, prompt, 5, model)
    for i, subclaim in enumerate(subclaims):
        subclaim["frequency-score"] = frequency_scores[i]

    accepted_subclaims = [
        subclaim for subclaim in subclaims if subclaim["frequency-score"] > threshold
    ]
    merged_output = merge_subclaims(client, accepted_subclaims, model, prompt)

    return merged_output, (accepted_subclaims, subclaims)


def get_subclaims(
    client,
    prompt,
    output,
    model,
    breakdown_prompt=BREAKDOWN_PROMPT,
    max_tokens=1000,
    temperature=0,
):
    """
    Takes in an output text and breaks it down into a list of sub-claims.
    """
    # Break into sub-claims.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant whose job is to break down your inputs into a set of small claims so that a human can easily check each one. Make sure that each claim is small and non-overlapping.",
        },
        {
            "role": "user",
            "content": ORIGINAL_PROMPT + prompt
        },
        {
            "role": "user",
            "content": breakdown_prompt + output,
        },
    ]
    # TODO: use query_model function here instead.
    completion = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    output = completion.choices[0].message.content
    output = output.replace("```jsonl\n", "")
    output = output.replace("\\", "\\\\")
    subclaims = output.replace("```", "")

    # Parse as jsonl.
    try:
        subclaims = [json.loads(line) for line in subclaims.splitlines() if line]
        return subclaims
    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        print(subclaims)
        return None


def get_frequency_scores(client, subclaims, prompt, n_samples, model):
    """
    Returns a vector of (frequency) scores corresponding to each entry of the subclaims list.
    """
    # Generate n_samples alternate outputs with temperature 1.0.
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        n=n_samples,
    )
    alternate_outputs = [choice.message.content for choice in completion.choices]
    claim_string = "\n".join(
        [str(i) + ": " + fact["subclaim"] for i, fact in enumerate(subclaims)]
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
        output = query_model(
            client, counting_prompt, model, max_tokens=1000, temperature=0
        )
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

    return final_scores


def default_merge_prompt(subclaims, prompt):
    claim_string = "\n".join(
        [str(i) + ": " + subclaim["subclaim"] for i, subclaim in enumerate(subclaims)]
    )
    return f"You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and try to use all facts as long as its possible. If no facts are given, reply to the instruction incorporating the fact that you dont know enough to fully respond. \n\nThe facts:\n{claim_string}\n\nThe instruction:\n{prompt}"


def get_truth(client, model, prompt, subclaims):
    claim_string = "\n".join(
        [str(i) + ": " + subclaim["subclaim"] for i, subclaim in enumerate(subclaims)]
    )
    message = VALIDITY_PROMPT + "\n The original prompt was: " + prompt + "\n The claims were: " + claim_string

    output = query_model(client, message, model)

    output = output.replace("```jsonl\n", "")
    output = output.replace("\\", "\\\\")
    output = output.replace("```", "")

    # Parse as jsonl.
    try:
        truth_scores = [json.loads(line) for line in output.splitlines() if line]
        return truth_scores
    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        print(output)
        return None


def merge_subclaims(
    client, subclaims, model, prompt, create_merge_prompt=default_merge_prompt
):
    """
    Takes in a list of sub-claims like [{'subclaim': 'Percy Liang is a computer scientist.', 'score': 5.0}, ...] and produces a merged output.
    """
    prompt = create_merge_prompt(subclaims, prompt)
    output = (
        query_model(client, prompt, model, max_tokens=1000, temperature=0)
        if subclaims
        else "Abstain."
    )
    return output
