import numpy as np
import requests

from typing import Dict
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

WIKIDATA_URL = "https://www.wikidata.org/w/api.php"

def get_id(response : Dict) -> str:
    wikidata_codes = list(response['entities'].keys())
    assert len(wikidata_codes) == 1
    return wikidata_codes[0]


def is_human(response : Dict, id: str) -> bool:
    instances = response['entities'][id]['claims'].get('P31', [])
    for inst in instances:
        if inst['mainsnak']['datavalue']['value']['id'] == 'Q5':
            return True
    return False

def validate_entity(k):
    name = k.split('/')[-1]
    response = requests.get(url=WIKIDATA_URL, params={"action" : "wbgetentities",
                                                    "sites" : "enwiki",
                                                    "titles" : name,
                                                    "normalize": "1",
                                                    "languages": "en",
                                                    "format": "json",
                                                    "props": "claims"})
    response = response.json()

    wiki_id = get_id(response)

    try:
        human = is_human(response, wiki_id)
    except:
        print(f"Failed {name}.")
        return name, False
    return name, human


if __name__ == "__main__":
    wiki_entities = np.load('/Users/cherian/Projects/pretraining_entities/wikipedia_entity_map.npz')
    with ThreadPoolExecutor(max_workers=25) as executor:
        res = list(
            tqdm(
                executor.map(
                    lambda k : validate_entity(k),
                    wiki_entities.keys()
                ),
                total=len(wiki_entities)
            )
        )

    import IPython; IPython.embed()