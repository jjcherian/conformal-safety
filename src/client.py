import os
import pickle
import time

from typing import Any, List

class Client:
    """
    Wrapper class for language models that we query. It keeps a cache of prompts and
    responses so that we don't have to requery things in experiments.
    """

    def __init__(self, cache_file, model : str = 'gpt-3.5-turbo'):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = model
        self.modified_cache = False

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def query(
            self, 
            prompt : str, 
            sample_idx : int = 0, 
            **kwargs
        ):
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if self.model is None:
            self.load_model()
        # print("I didn't find a cached copy!")
        output = self._query(prompt, **kwargs)

        return output
    
    def cache_outputs(
            self,
            prompts : List[str],
            sample_indices : List[int],
            outputs : List[Any]
    ):
        for prompt, sample_idx, output in zip(prompts, sample_indices, outputs):
            prompt = prompt.strip()
            cache_key = f"{prompt}_{sample_idx}"
            self.cache_dict[cache_key] = output
            self.modified_cache = True

    def save_cache(self):
        if self.modified_cache == False:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception: # if there are concurent processes, things can fail
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)  
        elif 's3' in self.cache_file:
            from aws_utils import s3_open
            s3_path = self.cache_file.removeprefix('s3://')
            bucket_name = s3_path.split('/')[0]
            path_to_file = '/'.join(s3_path.split('/')[1:])
            with s3_open(bucket_name, path_to_file) as fp:
                cache = pickle.load(fp)
        else:
            cache = {}
        return cache



