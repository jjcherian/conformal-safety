import argparse

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from dataset import load_dataset
from config import get_config

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
    

    

        


