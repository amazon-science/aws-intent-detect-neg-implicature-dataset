# There was actually a mistake that the input utterances from different datasets are not shuffled which will decrease the diversity of sub-sampled data.
# To that end, this program shuffles the data that are not yet predicted.
import json, random
# make sure reproducibility
random.seed(0)

with open("/Users/yuwzhan/Documents/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3.json", 'r') as f:
    data = json.load(f)

predicted = [d for d in data if 'prediction' in d]
rest = [d for d in data if 'prediction' not in d]

assert len(predicted) + len(rest) == len(data)

random.shuffle(rest)

data = predicted + rest
with open("/Users/yuwzhan/Documents/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_shuffled.json", 'w') as f:
    json.dump(data, f)

# TODO: check the output, make sure that all the data are there and the order is really different.