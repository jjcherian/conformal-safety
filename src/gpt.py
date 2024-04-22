import openai

from typing import List

from client import Client

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
class GPTClient(Client):
    def __init__(
            self, 
            cache_file : str,
            model : str = 'gpt-3.5-turbo'
    ):
        super(GPTClient, self).__init__(cache_file, model)
        self.client = openai.Client()
        self.tokens_used = 0
        self.requests_made = 0
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _query(
            self,
            prompt : List[str],
            role : List[str] = None,
            max_tokens : int = 1000,
            temperature: float = 0,
            response_format : str = None,
            n_samples: int = 1
    ):
        if role is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": role, "content": prompt}]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n_samples,
            logprobs=True
        )
        self.tokens_used += completion.usage.total_tokens
        self.requests_made += 1
        # print(self.tokens_used, self.requests_made)
        outputs = []
        for choice in completion.choices:
            output_dict = {
                'logprobs': choice.logprobs.content,
                'message': choice.message.content
            }
            outputs.append(output_dict)
        return outputs