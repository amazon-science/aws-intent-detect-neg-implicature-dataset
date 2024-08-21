"""
This code is for aggregating the results obtained over 3 runs.
"""
import os, json
from collections import defaultdict
import numpy as np

final_results = defaultdict(list)
for model_path in ["pretrain_data_paraphrase_comparative", "pretrain_data_paraphrase_comparative_seed33", "pretrain_data_paraphrase_comparative_seed666"]:
    with open(os.path.join(model_path, "averaged_logs_test.json"), 'r') as f:
        data = json.load(f)
    for k in data:
        final_results[k].append(data[k])

for k in final_results:
    final_results[k] = np.mean(final_results[k])

with open("pretrain_data_paraphrase_comparative_avg_logs.json", 'w') as f:
    json.dump(final_results, f, indent=4)