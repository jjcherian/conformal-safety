import string
import numpy as np
import os
import json

from concurrent.futures import ThreadPoolExecutor
# import logging

# from tqdm import tqdm
# from factscore.abstain_detection import is_response_abstained
from retrieval import DocDB, Retrieval

class Scorer(object):

    def __init__(self,
                 client,
                 config,
                 model_name="retrieval+ChatGPT",
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm", "retrieval"]
        self.model_name = model_name
        self.client = client
        self.config = config

        self.data_dir = config.model.annotator.data_path
        self.cache_dir = config.model.annotator.retrieval_cache_path

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        # self.abstain_detection_type = abstain_detection_type

        # self.data_dir = data_dir
        # self.cache_dir = cache_dir
        # if not os.path.exists(cache_dir):
        #     os.makedirs(cache_dir)

        self.af_generator = None

    def save_cache(self):
        self.client.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()
        for k, v in self.db:
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"

        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        if name == "medlfqa":
            datasets = {}
            suffix = "_test_MedLFQA.jsonl"

            # dataset_dir = "/Users/cherian/Projects/OLAPH/MedLFQA"
            for path in os.listdir(self.data_dir):
                if "MedLFQA" not in path:
                    continue
                dataset_name = path[:-len(suffix)]
                with open(os.path.join(self.data_dir, path), 'r') as fp:
                    datasets[dataset_name] = [json.loads(line) for line in fp.readlines()]
                retrieval = {}
                for _, dataset in datasets.items():
                    for pt in dataset:
                        retrieval[pt['Question']] = {
                            'context': pt['Free_form_answer'],
                            'must_have': pt['Must_have'],
                            'nice_to_have': pt['Nice_to_have']
                        }
                self.retrieval[name] = retrieval
        
        else:
            db_cache_path = os.path.join(self.cache_dir, f"db-{name}.pkl")
            cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

            self.db[name] = DocDB(db_path=db_path, data_path=data_path, cache_path=db_cache_path)
            self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, retrieval_type="bm25", batch_size=self.batch_size)
            # if "npm" in self.model_name:
            #     cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            #     embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            #     self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
            #                          "npm-single",
            #                          cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def get_score(self,
                  topics,
                  generations,
                  atomic_facts,
                  gamma=10,
                  knowledge_source=None):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
            atomic_facts = [atomic_facts]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision = []
                for fact in facts:
                    decision.append(
                        self._get_score(topic, generation, fact, knowledge_source, decision)
                    )
                score = np.mean([d["is_supported"] for d in decision])
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/max(len(facts), 1))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                # if len(scores) % 10 == 0:
                #     self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atom, knowledge_source, prev_decisions = []):
        definition = f"Answer the question about {topic} based on the given context and your previous answers.\n\n"
        atom = atom.strip()
        if knowledge_source == "medlfqa":
            context = self.retrieval[knowledge_source][topic]['context']
        else:
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            context = ""
            for psg in reversed(passages):
                context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
        definition += context.strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        prompt = f"{definition.strip()}\n\n"
        for prev_decision in prev_decisions:
            prev_score = "True" if prev_decision["is_supported"] else "False"
            prompt += f"Previous input: {prev_decision['atom']}\nTrue or False? Output: {prev_score}\n"
        
        prompt += f"Input: {atom.strip()} True or False?\nOutput:"
        # output = [{'message': 'blah blah blah'}]
        output = self.client.query(prompt) 

        # if type(output[1])==np.ndarray:
        #     # when logits are available
        #     logits = np.array(output[1])
        #     assert logits.shape[0] in [32000, 32001]
        #     true_score = logits[5852]
        #     false_score = logits[7700]
        #     is_supported = true_score > false_score
        # else:
        # when logits are unavailable
        generated_answer = output[0]['message'].lower()
        if "true" in generated_answer or "false" in generated_answer:
            if "true" in generated_answer and "false" not in generated_answer:
                is_supported = True
            elif "false" in generated_answer and "true" not in generated_answer:
                is_supported = False
            else:
                is_supported = generated_answer.index("true") > generated_answer.index("false")
        else:
            is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

        if is_supported and "npm" in self.model_name:
            npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
            is_supported = npprob > 0.3

        decision = {"atom": atom, "is_supported": is_supported}

        return decision