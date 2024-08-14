"""
split dev set for implicature data
"""
import random, os
from collections import defaultdict

random.seed(0)

for dataset in ["BANKING77", "HWU64", "CLINC150"]:

    with open(f"/fsx/users/yuwzhan/codes/IntentSemanticEncoder/build_toolkit/results/{dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in", 'r') as f:
        utts = f.readlines()

    with open(f"/fsx/users/yuwzhan/codes/IntentSemanticEncoder/build_toolkit/results/{dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label", 'r') as f:
        labs = f.readlines()
    
    lab2utt = defaultdict(list)
    for u, l in zip(utts, labs):
        lab2utt[l].append(u)
    
    lab2utt_dev = {}
    lab2utt_test = {}
    for l in lab2utt:
        random.shuffle(lab2utt[l])
        mid = len(lab2utt[l]) // 2
        lab2utt_dev[l] = lab2utt[l][:mid]
        lab2utt_test[l] = lab2utt[l][mid:]

    output_dir = f"/fsx/users/yuwzhan/codes/IntentSemanticEncoder/build_toolkit/results/{dataset}/implicature_splitted"
    dev_dir = os.path.join(output_dir, "dev")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    utts_dev, labs_dev = [], []
    for l in lab2utt_dev:
        utts_dev.extend(lab2utt_dev[l])
        labs_dev.extend([l] * len(lab2utt_dev[l]))
    dev_dir = os.path.join(output_dir, "dev")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(dev_dir, "utterances_gpt-4-0613_in_context_implicature.in"), 'w') as f:
        f.writelines(utts_dev)
    with open(os.path.join(dev_dir, "utterances_gpt-4-0613_in_context_implicature_label"), 'w') as f:
        f.writelines(labs_dev)

    utts_test, labs_test = [], []
    for l in lab2utt_test:
        utts_test.extend(lab2utt_test[l])
        labs_test.extend([l] * len(lab2utt_test[l]))
    with open(os.path.join(test_dir, "utterances_gpt-4-0613_in_context_implicature.in"), 'w') as f:
        f.writelines(utts_test)
    with open(os.path.join(test_dir, "utterances_gpt-4-0613_in_context_implicature_label"), 'w') as f:
        f.writelines(labs_test)