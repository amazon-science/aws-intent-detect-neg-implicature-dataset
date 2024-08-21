"""
This code will automatically extract errors in binary classification, and then print them out for analysis.
"""
import json

log_path = "results/pretrain_data_v7_paraphrase_252744_proc_disable_hard_pos_7_hard_neg_2/HWU64/binary_classification/implicature/logs.json"

if "HWU64" in log_path:
    dataset = "HWU64"
elif "CLINC150" in log_path:
    dataset = "CLINC150"
elif "BANKING77" in log_path:
    dataset = "BANKING77"

if "implicature" in log_path:
    data_path = f"../build_toolkit/results/{dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in"
    lab_path = f"../build_toolkit/results/{dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label"
elif "negated" in log_path:
    data_path = f"../build_toolkit/results/{dataset}/negation/modified_utterance_gpt-4-0613_negate_intent_all_v2.in"
    lab_path = f"../datasets/{dataset}/test/label"
elif "original" in log_path:
    data_path = f"../datasets/{dataset}/test/seq.in"
    lab_path = f"../datasets/{dataset}/test/label"

with open(log_path, 'r') as f:
    ranks = json.load(f)['rank']
with open(data_path, 'r') as f:
    utts = [u.strip() for u in f.readlines()]
with open(lab_path, 'r') as f:
    intents = [i.strip() for i in f.readlines()]

errors = []
for rank, utt, intent in zip(ranks, utts, intents):
    if rank == [1, 0]:
        errors.append((utt, intent))

breakpoint()
pass