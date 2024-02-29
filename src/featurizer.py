import numpy as np
import json


# def get_frequency(client, subclaims, prompt, n_samples, model):
#     """
#     Returns a vector of (frequency) scores corresponding to each entry of the subclaims list.
#     """
#     # Generate n_samples alternate outputs with temperature 1.0.
#     messages = [{"role": "user", "content": prompt}]
#     completion = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=1.0,
#         n=n_samples,
#     )
#     alternate_outputs = [choice.message.content for choice in completion.choices]
#     claim_string = "\n".join(
#         [str(i) + ": " + fact["subclaim"] for i, fact in enumerate(subclaims)]
#     )

#     # Count the number of times the alternate outputs support the sub-claims (using LM).
#     # TODO: should this really be -1, 0, 1? Before it was 0, 1.
#     final_scores = [0.0] * len(subclaims)
#     for output in alternate_outputs:
#         counting_prompt = (
#             'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "score":[SCORE]}. Directly return the jsonl with no explanation or other formatting. For the [SCORE], return 1 for supports, -1 for contradicts, and 0 for unrelated. The claims are:\n'
#             + claim_string
#             + "\n\nThe text is:\n"
#             + output
#         )
#         output = query_model(
#             client, counting_prompt, model, max_tokens=1000, temperature=0
#         )
#         output = output.replace("```jsonl\n", "")
#         output = output.replace("```", "")
#         try:
#             for i, line in enumerate(output.splitlines()):
#                 scores = json.loads(line)
#                 idx = int(scores["id"])
#                 final_scores[idx] += float(scores["score"])
#         except Exception as ex:
#             print(ex)
#             print("Failed to parse as jsonl")
#             print(output)

#     return final_scores