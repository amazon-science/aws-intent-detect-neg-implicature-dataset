# average pretrain results for a specific model on test set
import argparse
import json, os
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True,
                    help="provide a absolute path to the model, where the evaluation results are produced")
parser.add_argument("--split", type=str, default="dev")
args = parser.parse_args()

model_name=args.model_name

avg_logs = {}
dataset_logs = defaultdict(dict)

# ----- triplet task negation -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/triplet_task_{args.split}set/negation/logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (explicit)'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (explicit)'] = logs['accuracy_t2']
avg_logs['T1 (explicit)'] = np.mean(acc_t1)
avg_logs['T2 (explicit)'] = np.mean(acc_t2)

# ----- triplet task implicature -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/triplet_task_{args.split}set/implicature/utterances_gpt-4-0613_in_context_implicature_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (implicature)'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (implicature)'] = logs['accuracy_t2']
avg_logs['T1 (implicature)'] = np.mean(acc_t1)
avg_logs['T2 (implicature)'] = np.mean(acc_t2)

# ----- binary classification original -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/binary_classification_{args.split}set/original/logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_original'] = logs['accuracy']
avg_logs['binary_classification_original'] = np.mean(acc)

# ----- binary classification negated -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/binary_classification_{args.split}set/negated/logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_negated'] = logs['accuracy']
avg_logs['binary_classification_negated'] = np.mean(acc)

# ----- binary classification implicature -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/binary_classification_{args.split}set/implicature/logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_implicature'] = logs['accuracy']
avg_logs['binary_classification_implicature'] = np.mean(acc)

# ----- clustering original -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/clustering_{args.split}set/original/logs.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_original_kmeans'] = logs['kmeans']
        dataset_logs[dataset]['clustering_original_agg']= logs['agglomerative']
avg_logs['clustering_original_kmeans'] = np.mean(kmeans)
avg_logs['clustering_original_agg'] = np.mean(agg)

# ----- clustering implicature -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/clustering_{args.split}set/implicature/logs.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_implicature_kmeans'] = logs['kmeans']
        dataset_logs[dataset]['clustering_implicature_agg'] = logs['agglomerative']
avg_logs['clustering_implicature_kmeans'] = np.mean(kmeans)
avg_logs['clustering_implicature_agg'] = np.mean(agg)

# ----- classification implicature -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/protonet_fullwayclass_{args.split}set/implicature/0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_0shot'] = logs['accuracy']
    with open(os.path.join(f"{model_name}/{dataset}/protonet_fullwayclass_{args.split}set/implicature/10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_10shot'] = logs['accuracy']
avg_logs['classification_implicature_0shot'] = np.mean(acc_0shot)
avg_logs['classification_implicature_10shot'] = np.mean(acc_10shot)

# ----- classification original -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{model_name}/{dataset}/protonet_fullwayclass_{args.split}set/original/0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_0shot'] = logs['accuracy']
    with open(os.path.join(f"{model_name}/{dataset}/protonet_fullwayclass_{args.split}set/original/10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_10shot'] = logs['accuracy']
avg_logs['classification_original_0shot'] = np.mean(acc_0shot)
avg_logs['classification_original_10shot'] = np.mean(acc_10shot)

print(avg_logs)

with open(f"{model_name}/averaged_logs_{args.split}.json", 'w') as f:
    json.dump(avg_logs, f, indent=4)

with open(f"{model_name}/dataset_logs_{args.split}.json", 'w') as f:
    json.dump(dataset_logs, f, indent=4)