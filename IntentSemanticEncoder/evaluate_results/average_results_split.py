import json, os
import numpy as np
from collections import defaultdict

# split="dev"
split="test"

avg_logs = {}
dataset_logs = defaultdict(dict)

# ----- triplet task negation IAE -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/negation/iae_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (explicit)'] = {'IAE': logs['accuracy_t1']}
        dataset_logs[dataset]['T2 (explicit)'] = {'IAE': logs['accuracy_t2']}
avg_logs['T1 (explicit)'] = {'IAE': np.mean(acc_t1)}
avg_logs['T2 (explicit)'] = {'IAE': np.mean(acc_t2)}

# ----- triplet task negation paraphrase -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/negation/paraphrase_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (explicit)']['paraphrase'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (explicit)']['paraphrase'] = logs['accuracy_t2']
avg_logs['T1 (explicit)']['paraphrase'] = np.mean(acc_t1)
avg_logs['T2 (explicit)']['paraphrase'] = np.mean(acc_t2)

# ----- triplet task negation Instructor-base -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/negation/instructor_base_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (explicit)']['instructor-base'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (explicit)']['instructor-base'] = logs['accuracy_t2']
avg_logs['T1 (explicit)']['instructor-base'] = np.mean(acc_t1)
avg_logs['T2 (explicit)']['instructor-base'] = np.mean(acc_t2)

# ----- triplet task negation Instructor-large -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/negation/instructor_large_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (explicit)']['instructor-large'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (explicit)']['instructor-large'] = logs['accuracy_t2']
avg_logs['T1 (explicit)']['instructor-large'] = np.mean(acc_t1)
avg_logs['T2 (explicit)']['instructor-large'] = np.mean(acc_t2)

# ----- triplet task implicature IAE -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/implicature/iae_utterances_gpt-4-0613_in_context_implicature_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (implicature)'] = {'IAE': logs['accuracy_t1']}
        dataset_logs[dataset]['T2 (implicature)'] = {'IAE': logs['accuracy_t2']}
avg_logs['T1 (implicature)'] = {'IAE': np.mean(acc_t1)}
avg_logs['T2 (implicature)'] = {'IAE': np.mean(acc_t2)}

# ----- triplet task implicature paraphrase -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/implicature/paraphrase_utterances_gpt-4-0613_in_context_implicature_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (implicature)']['paraphrase'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (implicature)']['paraphrase'] = logs['accuracy_t2']
avg_logs['T1 (implicature)']['paraphrase'] = np.mean(acc_t1)
avg_logs['T2 (implicature)']['paraphrase'] = np.mean(acc_t2)

# ----- triplet task implicature Instructor-base -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/implicature/instructor_base_utterances_gpt-4-0613_in_context_implicature_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (implicature)']['instructor-base'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (implicature)']['instructor-base'] = logs['accuracy_t2']
avg_logs['T1 (implicature)']['instructor-base'] = np.mean(acc_t1)
avg_logs['T2 (implicature)']['instructor-base'] = np.mean(acc_t2)

# ----- triplet task implicature Instructor-large -----
acc_t1, acc_t2 = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/triplet_task_{split}set/implicature/instructor_large_utterances_gpt-4-0613_in_context_implicature_logs_seed_100.json")) as f:
        logs = json.load(f)
        acc_t1.append(logs['accuracy_t1'])
        acc_t2.append(logs['accuracy_t2'])
        dataset_logs[dataset]['T1 (implicature)']['instructor-large'] = logs['accuracy_t1']
        dataset_logs[dataset]['T2 (implicature)']['instructor-large'] = logs['accuracy_t2']
avg_logs['T1 (implicature)']['instructor-large'] = np.mean(acc_t1)
avg_logs['T2 (implicature)']['instructor-large'] = np.mean(acc_t2)

# ----- binary classification original IAE -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/original/iae_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_original'] = {'IAE': logs['accuracy']}
avg_logs['binary_classification_original'] = {'IAE': np.mean(acc)}

# ----- binary classification original paraphrase -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/original/paraphrase_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_original']['paraphrase'] = logs['accuracy']
avg_logs['binary_classification_original']['paraphrase'] = np.mean(acc)

# ----- binary classification original instructor-base -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/original/instructor_base_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_original']['instructor-base'] = logs['accuracy']
avg_logs['binary_classification_original']['instructor-base'] = np.mean(acc)

# ----- binary classification original instructor-large -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/original/instructor_large_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_original']['instructor-large'] = logs['accuracy']
avg_logs['binary_classification_original']['instructor-large'] = np.mean(acc)

# ----- binary classification negated IAE -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/negated/iae_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_negated'] = {'IAE': logs['accuracy']}
avg_logs['binary_classification_negated'] = {'IAE': np.mean(acc)}

# ----- binary classification negated paraphrase -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/negated/paraphrase_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_negated']['paraphrase'] = logs['accuracy']
avg_logs['binary_classification_negated']['paraphrase'] = np.mean(acc)

# ----- binary classification negated instructor-base -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/negated/instructor_base_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_negated']['instructor-base'] = logs['accuracy']
avg_logs['binary_classification_negated']['instructor-base'] = np.mean(acc)

# ----- binary classification negated instructor-large -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/negated/instructor_large_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_negated']['instructor-large'] = logs['accuracy']
avg_logs['binary_classification_negated']['instructor-large'] = np.mean(acc)

# ----- binary classification implicature IAE -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/implicature/iae_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_implicature'] = {'IAE': logs['accuracy']}
avg_logs['binary_classification_implicature'] = {'IAE': np.mean(acc)}

# ----- binary classification implicature paraphrase -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/implicature/paraphrase_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_implicature']['paraphrase'] = logs['accuracy']
avg_logs['binary_classification_implicature']['paraphrase'] = np.mean(acc)

# ----- binary classification implicature instructor-base -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/implicature/instructor_base_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_implicature']['instructor-base'] = logs['accuracy']
avg_logs['binary_classification_implicature']['instructor-base'] = np.mean(acc)

# ----- binary classification implicature instructor-large -----
acc = []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/binary_classification_{split}set/implicature/instructor_large_logs.json")) as f:
        logs = json.load(f)
        acc.append(logs['accuracy'])
        dataset_logs[dataset]['binary_classification_implicature']['instructor-large'] = logs['accuracy']
avg_logs['binary_classification_implicature']['instructor-large'] = np.mean(acc)

# ----- clustering original IAE -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/original/iae.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_original_kmeans'] = {'IAE': logs['kmeans']}
        dataset_logs[dataset]['clustering_original_agg'] = {'IAE': logs['agglomerative']}
avg_logs['clustering_original_kmeans'] = {'IAE': np.mean(kmeans)}
avg_logs['clustering_original_agg'] = {'IAE': np.mean(agg)}

# ----- clustering original paraphrase -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/original/paraphrase.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_original_kmeans']['paraphrase'] = logs['kmeans']
        dataset_logs[dataset]['clustering_original_agg']['paraphrase'] = logs['agglomerative']
avg_logs['clustering_original_kmeans']['paraphrase'] = np.mean(kmeans)
avg_logs['clustering_original_agg']['paraphrase'] = np.mean(agg)

# ----- clustering original instructor-base -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/original/instructor_base.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_original_kmeans']['instructor-base'] = logs['kmeans']
        dataset_logs[dataset]['clustering_original_agg']['instructor-base'] = logs['agglomerative']
avg_logs['clustering_original_kmeans']['instructor-base'] = np.mean(kmeans)
avg_logs['clustering_original_agg']['instructor-base'] = np.mean(agg)

# ----- clustering original instructor-large -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/original/instructor_large.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_original_kmeans']['instructor-large'] = logs['kmeans']
        dataset_logs[dataset]['clustering_original_agg']['instructor-large'] = logs['agglomerative']
avg_logs['clustering_original_kmeans']['instructor-large'] = np.mean(kmeans)
avg_logs['clustering_original_agg']['instructor-large'] = np.mean(agg)

# ----- clustering implicature IAE -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/implicature/iae.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_implicature_kmeans'] = {'IAE': logs['kmeans']}
        dataset_logs[dataset]['clustering_implicature_agg'] = {'IAE': logs['agglomerative']}
avg_logs['clustering_implicature_kmeans'] = {'IAE': np.mean(kmeans)}
avg_logs['clustering_implicature_agg'] = {'IAE': np.mean(agg)}

# ----- clustering implicature paraphrase -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/implicature/paraphrase.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_implicature_kmeans']['paraphrase'] = logs['kmeans']
        dataset_logs[dataset]['clustering_implicature_agg']['paraphrase'] = logs['agglomerative']
avg_logs['clustering_implicature_kmeans']['paraphrase'] = np.mean(kmeans)
avg_logs['clustering_implicature_agg']['paraphrase'] = np.mean(agg)

# ----- clustering implicature instructor-base -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/implicature/instructor_base.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_implicature_kmeans']['instructor-base'] = logs['kmeans']
        dataset_logs[dataset]['clustering_implicature_agg']['instructor-base'] = logs['agglomerative']
avg_logs['clustering_implicature_kmeans']['instructor-base'] = np.mean(kmeans)
avg_logs['clustering_implicature_agg']['instructor-base'] = np.mean(agg)

# ----- clustering implicature instructor-large -----
kmeans, agg = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/clustering_{split}set/implicature/instructor_large.json")) as f:
        logs = json.load(f)
        kmeans.append(logs['kmeans'])
        agg.append(logs['agglomerative'])
        dataset_logs[dataset]['clustering_implicature_kmeans']['instructor-large'] = logs['kmeans']
        dataset_logs[dataset]['clustering_implicature_agg']['instructor-large'] = logs['agglomerative']
avg_logs['clustering_implicature_kmeans']['instructor-large'] = np.mean(kmeans)
avg_logs['clustering_implicature_agg']['instructor-large'] = np.mean(agg)

# ----- classification implicature IAE -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/iae_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_0shot'] = {'IAE': logs['accuracy']}
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/iae_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_10shot'] = {'IAE': logs['accuracy']}
avg_logs['classification_implicature_0shot'] = {'IAE': np.mean(acc_0shot)}
avg_logs['classification_implicature_10shot'] = {'IAE': np.mean(acc_10shot)}

# ----- classification implicature paraphrase -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/paraphrase_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_0shot']['paraphrase'] = logs['accuracy']
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/paraphrase_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_10shot']['paraphrase'] = logs['accuracy']
avg_logs['classification_implicature_0shot']['paraphrase'] = np.mean(acc_0shot)
avg_logs['classification_implicature_10shot']['paraphrase'] = np.mean(acc_10shot)

# ----- classification implicature instructor-base -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/instructor_base_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_0shot']['instructor-base'] = logs['accuracy']
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/instructor_base_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_10shot']['instructor-base'] = logs['accuracy']
avg_logs['classification_implicature_0shot']['instructor-base'] = np.mean(acc_0shot)
avg_logs['classification_implicature_10shot']['instructor-base'] = np.mean(acc_10shot)

# ----- classification implicature instructor-large -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/instructor_large_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_0shot']['instructor-large'] = logs['accuracy']
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/implicature/instructor_large_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_implicature_10shot']['instructor-large'] = logs['accuracy']
avg_logs['classification_implicature_0shot']['instructor-large'] = np.mean(acc_0shot)
avg_logs['classification_implicature_10shot']['instructor-large'] = np.mean(acc_10shot)

# ----- classification original IAE -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/iae_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_0shot'] = {'IAE': logs['accuracy']}
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/iae_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_10shot'] = {'IAE': logs['accuracy']}
avg_logs['classification_original_0shot'] = {'IAE': np.mean(acc_0shot)}
avg_logs['classification_original_10shot'] = {'IAE': np.mean(acc_10shot)}

# ----- classification original paraphrase -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/paraphrase_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_0shot']['paraphrase'] = logs['accuracy']
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/paraphrase_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_10shot']['paraphrase'] = logs['accuracy']
avg_logs['classification_original_0shot']['paraphrase'] = np.mean(acc_0shot)
avg_logs['classification_original_10shot']['paraphrase'] = np.mean(acc_10shot)

# ----- classification original instructor-base -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/instructor_base_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_0shot']['instructor-base'] = logs['accuracy']
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/instructor_base_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_10shot']['instructor-base'] = logs['accuracy']
avg_logs['classification_original_0shot']['instructor-base'] = np.mean(acc_0shot)
avg_logs['classification_original_10shot']['instructor-base'] = np.mean(acc_10shot)

# ----- classification original instructor-large -----
acc_0shot, acc_10shot = [], []
for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/instructor_large_0shots_logs.json")) as f:
        logs = json.load(f)
        acc_0shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_0shot']['instructor-large'] = logs['accuracy']
    with open(os.path.join(f"{dataset}/protonet_fullwayclass_{split}set/original/instructor_large_10shots_logs.json")) as f:
        logs = json.load(f)
        acc_10shot.append(logs['accuracy'])
        dataset_logs[dataset]['classification_original_10shot']['instructor-large'] = logs['accuracy']
avg_logs['classification_original_0shot']['instructor-large'] = np.mean(acc_0shot)
avg_logs['classification_original_10shot']['instructor-large'] = np.mean(acc_10shot)

print(avg_logs)

with open(f"averaged_logs_{split}.json", 'w') as f:
    json.dump(avg_logs, f, indent=4)

with open(f"dataset_logs_{split}.json", 'w') as f:
    json.dump(dataset_logs, f, indent=4)