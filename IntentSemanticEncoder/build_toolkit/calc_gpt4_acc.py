"""
This code is used for calculating intent alignment accuracy
"""
import json

acc = {}
for dataset in ["BANKING77", "HWU64", "CLINC150"]:
    with open(f"results/{dataset}/intent_align/gpt-4-0613_intent_alignment_v1.json", 'r') as f:
        data = json.load(f)
    acc[dataset] = sum([1 for d in data if d['lab'] == d['prediction']]) / len(data)

average = sum([acc[d] for d in acc]) / len(acc)
acc['average'] = average

with open("intent_alignment_logs.json", 'w') as f:
    json.dump(acc, f)