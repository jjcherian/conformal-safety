import logging
import numpy as np
import requests

from typing import Dict
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

ENTITY_PATH = '/data/jcherian/wikipedia_entity_map.npz'
WIKIDATA_URL = "https://www.wikidata.org/w/api.php"
logger = logging.getLogger(__name__)
logging.basicConfig(filename='human.log', level=logging.INFO)


def get_id(response : Dict) -> str:
    if response.get("entities", None) is None:
        return None
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
    adapter = requests.adapters.HTTPAdapter(max_retries=10)
    with requests.session() as s:
        s.mount("https://", adapter)
        response = s.get(url=WIKIDATA_URL, params={"action" : "wbgetentities",
                                                        "sites" : "enwiki",
                                                        "titles" : name,
                                                        "normalize": "1",
                                                        "languages": "en",
                                                        "format": "json",
                                                        "props": "claims"})

    try:
        response = response.json()
    except:
        print(response.text)

    wiki_id = get_id(response)
    
    if wiki_id is None:
        return name, False

    try:
        human = is_human(response, wiki_id)
    except:
        return name, False
    logger.info(f"{name}, {human}")
    return name, human


if __name__ == "__main__":
    wiki_entities = np.load(ENTITY_PATH)
    entity_names = list(wiki_entities.keys())
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            res = list(
                tqdm(
                    executor.map(
                        lambda k : validate_entity(k),
                        entity_names
                    ),
                    total=len(entity_names)
                )
            )
    except:
        import pickle
        with open('human.pkl', 'wb') as fp:
            pickle.dump(res, fp)
    

    import pickle
    with open('human.pkl', 'wb') as fp:
        pickle.dump(res, fp)

    import IPython; IPython.embed()
