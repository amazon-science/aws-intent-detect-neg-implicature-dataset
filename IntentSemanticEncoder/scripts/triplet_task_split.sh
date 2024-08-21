#!/bin/bash -x

# split="dev"
split="test"

# ===== negation =====

# !warning: positive is sampled from original data, no need to provide positive data

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/instructor_large_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/instructor_large_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-large" \
        --overwrite
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/instructor_base_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/instructor_base_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-base" \
        --overwrite
done

# ----- IAE -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/iae_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/iae_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "iae" \
        --model_name_or_path models/iae_model \
        --overwrite
done

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/paraphrase_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/negation/paraphrase_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
        --overwrite
done

# ===== implicature =====

# positive_file_name=utterances_gpt-3.5-turbo-0613_generate_implicature_v1
positive_file_name=utterances_gpt-4-0613_in_context_implicature

# !warning: notice that since positive data created by llm might not align to original data, we still need --sample_positive

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    positive_data_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}.in
    positive_data_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}_label
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/instructor_large_${positive_file_name}_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/instructor_large_${positive_file_name}_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --positive_data_seqin $positive_data_seqin \
        --positive_data_label $positive_data_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-large" \
        --sample_positive \
        --overwrite
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    positive_data_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}.in
    positive_data_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}_label
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/instructor_base_${positive_file_name}_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/instructor_base_${positive_file_name}_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --positive_data_seqin $positive_data_seqin \
        --positive_data_label $positive_data_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-base" \
        --sample_positive \
        --overwrite
done

# ----- IAE -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    positive_data_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}.in
    positive_data_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}_label
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/iae_${positive_file_name}_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/iae_${positive_file_name}_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --positive_data_seqin $positive_data_seqin \
        --positive_data_label $positive_data_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "iae" \
        --model_name_or_path models/iae_model \
        --sample_positive \
        --overwrite
done

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    original_data_path=datasets/${dataset}/${split}
    negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    positive_data_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}.in
    positive_data_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}_label
    save_emb_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/paraphrase_${positive_file_name}_embeds_seed_100.hdf5
    save_log_path=evaluate_results/${dataset}/triplet_task_${split}set/implicature/paraphrase_${positive_file_name}_logs_seed_100.json
    python evaluate_code/triplet_task.py \
        --original_data_path $original_data_path \
        --negative_data_path $negative_data_path \
        --positive_data_seqin $positive_data_seqin \
        --positive_data_label $positive_data_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
        --sample_positive \
        --overwrite
done