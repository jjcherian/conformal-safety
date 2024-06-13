import argparse
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from dataset import load_dataset
from config import get_config

from featurizer import get_frequency, get_self_eval, get_bool_eval
from gpt import GPTClient

def parse_args():
    parser = argparse.ArgumentParser(
        prog="conformal-safety",
        description="Auto-filter claims from LLM to meet accuracy and safety guarantees.",
    )
    parser.add_argument('-config_path', '-c', default='configs/default.toml', help="Config for construction.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.config_path)

    dataset = load_dataset(config)

    # client = GPTClient(f'.cache/{config.dataset.name}_frequency.pkl')


    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     frequencies = list(
    #         tqdm(
    #             executor.map(
    #                 lambda x: get_frequency(client, [af['atom'] for af in x['atomic_facts']], x['prompt'], config.model.prob.frequency.model),
    #                 dataset
    #             ),
    #             total=len(dataset)
    #         )
    #     )
    # client.save_cache()

    # eval_client = GPTClient(f'.cache/{config.dataset.name}_self_evals.pkl')

    # with ThreadPoolExecutor(max_workers=25) as executor:
    #     self_evals = list(
    #         tqdm(
    #             executor.map(
    #                 lambda x: get_self_eval(x['prompt'], [af['atom'] for af in x['atomic_facts']], eval_client),
    #                 dataset
    #             ),
    #             total=len(dataset)
    #         )
    #     )
    # eval_client.save_cache()

    # bool_client = GPTClient(f'.cache/{config.dataset.name}_bool_evals.pkl')

    # with ThreadPoolExecutor(max_workers=25) as executor:
    #     self_bools = list(
    #         tqdm(
    #             executor.map(
    #                 lambda x: get_bool_eval(x['prompt'], [af['atom'] for af in x['atomic_facts']], bool_client),
    #                 dataset
    #             ),
    #             total=len(dataset)
    #         )
    #     )
    # bool_client.save_cache()

    # features = np.concatenate(
    #     [
    #         np.concatenate(frequencies).reshape(-1,1),
    #         np.concatenate(self_evals).reshape(-1,1)
    #     ],
    #     axis=1
    # )

    import IPython; IPython.embed()
    

    

        


