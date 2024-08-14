# This code might be an artifact, merge the generated and retrieved data together
# The problem is that I have to run generation and retrieval in parallel in order to finish them
# in a limited time.
import json
from copy import deepcopy

# uncomment one block when you are ready
# ----- instructor-base -----
# files_to_merge = [
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative_ret_instructor-base.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive.json"
# ]
# ----- instructor-large -----
# files_to_merge = [
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative_ret_instructor-large.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive.json"
# ]
# ----- paraphrase -----
# files_to_merge = [
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative_ret_paraphrase-mpnet-base-v2.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive.json"
# ]

# below deprecated
# ----- iae -----
# files_to_merge = [
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative_ret_iae_model.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json",
#     "/efs/yuwzhan/codes/IntentSemanticEncoder/build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive.json"
# ]

data_to_merge = []
for filename in files_to_merge:
    with open(filename, 'r') as f:
        data_to_merge.append(json.load(f))

final_data = []
for data_list in zip(*data_to_merge):
    # guarantee they are the same data
    assert len(set([d['utt'] for d in data_list])) == 1
    final_datum = {}
    for d in data_list:
        for k in d:
            if k not in final_datum:
                final_datum[k] = d[k]
    final_data.append(final_datum)

# always get the same number of data as retrieved_negative
final_data = [d for d in final_data if 'retrieved_negative' in d]
print(len(final_data))

# TODO: In order to discriminate the versions of pre-training data, I have to add `vi` into file names.
# So please remember to manually change them into the names you want.
# with open(f"results/pretrain/pretrain_data_v1_{len(final_data)}.json", 'w') as f:
# with open(f"results/pretrain/pretrain_data_v2_{len(final_data)}.json", 'w') as f:
# with open(f"results/pretrain/pretrain_data_v3_instructor_base_{len(final_data)}.json", 'w') as f:
# with open(f"results/pretrain/pretrain_data_v4_instructor_large_{len(final_data)}.json", 'w') as f:
# with open(f"results/pretrain/pretrain_data_v5_paraphrase_{len(final_data)}.json", 'w') as f:
# with open(f"results/pretrain/pretrain_data_v6_iae_{len(final_data)}.json", 'w') as f:
# with open(f"results/pretrain/pretrain_data_v7_paraphrase_{len(final_data)}.json", 'w') as f:
# with open(f"results/pretrain/pretrain_data_v8_iae_{len(final_data)}.json", 'w') as f:
    json.dump(final_data, f, indent=4)