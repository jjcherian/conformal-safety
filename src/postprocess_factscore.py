import json

output = []
prompt_to_idx = {}
idx = 0
with open("/Users/cherian/Downloads/factscore-unlabeled-predictions/ChatGPT.jsonl") as fp:
    for line in fp:
        res = json.loads(line)
        new_res = {}
        new_res['prompt'] = res['prompt']
        new_res['claims'] = []
        annotator = 'ChatGPT_Labels' if 'ChatGPT_Labels' in res else 'LLAMA+NP_Labels'
        for fact, annotation in zip(res['facts'], res[annotator]):
            a = 'T' if annotation == 'S' else 'F'
            new_res['claims'].append(
                {'message': fact, 'annotation': a}
            )
        output.append(new_res)
        prompt_to_idx[res['prompt']] = idx
        idx += 1

with open("/Users/cherian/Projects/FActScore/factscore/data/unlabeled/ChatGPT.jsonl", 'r') as fp:
    for line in fp:
        res = json.loads(line)
        idx = prompt_to_idx.get(res['input'], None)
        if idx is None:
            continue
        else:
            output[idx]['response'] = res['output']
            output[idx]['topic'] = res['topic']
            output[idx]['metadata'] = res['cat']

with open("data/factscore_processed.json", 'w') as fp:
    fp.write(json.dumps(output) + "\n")