"""
This code is used to get csv file for all the results
"""
import csv, json, os

models = [
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_llm",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_positive_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_negative_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_4_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_5_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_6_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_hard_neg_2_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_hard_neg_3_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_hard_neg_4_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_hard_neg_2_4_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_4_hard_neg_2_4_lr",
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_4_5_hard_neg_2_4_lr"
]

result_file_name = "averaged_logs_dev.json"

metrics = [
    "T1 (explicit)",
    "T2 (explicit)",
    "T1 (implicature)",
    "T2 (implicature)",
    "binary_classification_original",
    "binary_classification_negated",
    "binary_classification_implicature",
    "clustering_original_kmeans",
    "clustering_original_agg",
    "clustering_implicature_kmeans",
    "clustering_implicature_agg",
    "classification_original_0shot",
    "classification_original_10shot",
    "classification_implicature_0shot",
    "classification_implicature_10shot",
]

results = []
for model in models:
    cur_results = [model]
    with open(os.path.join(model, result_file_name), 'r') as f:
        data = json.load(f)
        for metric in metrics:
            cur_results.append(round(data[metric] * 100, 2))
    results.append(cur_results)

with open("tables/instructor_large_dev.csv", 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Model'] + metrics)
    csv_writer.writerows(results)