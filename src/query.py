from typing import Dict, List
import openai

SUBCLAIM_PROMPT = 'Please breakdown the following response to a prompt into a set of small, independent claims. Return each subclaim (with no other characters) on a new line. \n'

MERGE_PROMPT = "You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and use ALL of the facts provided. If no facts are given, reply and say that you don't know enough to respond.\n"

ANNOTATION_PROMPT = 'You will get an instruction and a set of claims made in response to that instruction. Determine whether each claim is true, subjective, or false. Each returned determination should be {"claim_id": ID, "value": TRUTH_VALUE} and be on its own line with NO other characters. The truth value should be in quotes and it should be T for Factual, S for Subjective, and F for False.\n'

FREQUENCY_PROMPT = 'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "score":[SCORE]}. Directly return the jsonl with no explanation or other formatting. For the [SCORE], return 1 for supports, -1 for contradicts, and 0 for unrelated.\n'

def _concat_claims(
    subclaims : List[str]
) -> str:
    return "\n".join(
        f"{i}: {subclaim['message']}" for i, subclaim in enumerate(subclaims) 
    )

def generate_subclaim_prompt(
    prompt : str, 
    response : str
) -> str:
    final_output = SUBCLAIM_PROMPT + f"The original instruction was: {prompt}\n"
    final_output += f"The response to be broken down into subclaims is: {response}"

    return final_output

def generate_merge_prompt(
    prompt : str,
    subclaims : List[str]
) -> str:
    final_output = MERGE_PROMPT + f"The original instruction was: {prompt}\n"

    final_output += f"The facts are: {_concat_claims(subclaims)}"

    return final_output

def generate_annotation_prompt(
    prompt : str,
    subclaims : List[str]
) -> str:
    final_output = ANNOTATION_PROMPT + f"The original instruction was: {prompt}\n"
    final_output += f"The claims are: \n{_concat_claims(subclaims)}"

    return final_output

def generate_frequency_prompt(
    subclaims : List[str],
    output : str,
) -> str:
    final_output = FREQUENCY_PROMPT + f"The claims are: {_concat_claims(subclaims)}\n"
    final_output += f"The text is: {output}"
    return final_output

def query_gpt(
        client : openai.Client,
        prompts : List[str],
        model : str = "gpt-3.5-turbo",
        roles : List[str] = None,
        max_tokens : int = 1000,
        temperature: float = 0,
        response_format : str = None,
        n_samples: int = 1
):
    if roles is None:
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
    else:
        messages = [{"role": role, "content": prompt} for role, prompt in zip(roles, prompts)]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n_samples,
        logprobs=True
    )
    return completion

def query_embedding(
        client : openai.Client,
        prompts : List[str],
        model : str = "text-embedding-3-small",
        **kwargs
):
    embed = client.embeddings.create(input = prompts, model = model, **kwargs).data[0].embedding
    return embed

def query_llm(
        prompts : List[str],
        model : str,
        **kwargs
) -> Dict:
    if 'gpt' in model:
        client = openai.Client() # OPENAI_API_KEY should be set as an environment variable
        completion = query_gpt(client, prompts, model, **kwargs)
        outputs = []
        for choice in completion.choices:
            output_dict = {
                'logprobs': choice.logprobs.content,
                'message': choice.message.content
            }
            outputs.append(output_dict)
        return outputs
    elif 'embedding' in model:
        client = openai.Client()
        output = query_embedding(client, prompts, model, **kwargs)
        return output

    else:
        raise ValueError(f"Model {model} is not supported in query.")