#!/bin/bash -x

echo "!Deprecated, you should use \"triplet_task_v2.sh\""
exit 1

# ===== negation =====

# !warning: positive is sampled from original data, no need to provide positive data

# ----- instructor-large -----
dataset=BANKING77
negate_prompt_version=v4_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/instructor_large_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/instructor_large_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-large" \
    --overwrite

dataset=HWU64
negate_prompt_version=v1_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/instructor_large_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/instructor_large_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-large" \
    --overwrite

dataset=CLINC150
negate_prompt_version=v1
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/instructor_large_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/instructor_large_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-large" \
    --overwrite

# ----- instructor-base -----
dataset=BANKING77
negate_prompt_version=v4_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/instructor_base_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/instructor_base_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-base" \
    --overwrite

dataset=HWU64
negate_prompt_version=v1_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/instructor_base_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/instructor_base_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-base" \
    --overwrite

dataset=CLINC150
negate_prompt_version=v1
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/instructor_base_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/instructor_base_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-base" \
    --overwrite

# ----- IAE -----
dataset=BANKING77
negate_prompt_version=v4_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/iae_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/iae_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "iae" \
    --model_name_or_path models/iae_model \
    --overwrite

dataset=HWU64
negate_prompt_version=v1_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/iae_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/iae_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "iae" \
    --model_name_or_path models/iae_model \
    --overwrite

dataset=CLINC150
negate_prompt_version=v1
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
save_emb_path=evaluate_results/${dataset}/triplet_task/negation/iae_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/negation/iae_logs_seed_100.json
python evaluate_code/triplet_task.py \
    --original_data_path $original_data_path \
    --negative_data_path $negative_data_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "iae" \
    --model_name_or_path models/iae_model \
    --overwrite

# ===== implicature =====

# positive_file_name=utterances_gpt-3.5-turbo-0613_generate_implicature_v1
positive_file_name=utterances_gpt-4-0613_in_context_implicature

# !warning: notice that since positive data created by llm might not align to original data, we still need --sample_positive

# ----- instructor-large -----
dataset=BANKING77
negate_prompt_version=v4_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_large_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_large_${positive_file_name}_logs_seed_100.json
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

dataset=HWU64
negate_prompt_version=v1_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_large_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_large_${positive_file_name}_logs_seed_100.json
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

dataset=CLINC150
negate_prompt_version=v1
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_large_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_large_${positive_file_name}_logs_seed_100.json
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

# ----- instructor-base -----
dataset=BANKING77
negate_prompt_version=v4_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_base_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_base_${positive_file_name}_logs_seed_100.json
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

dataset=HWU64
negate_prompt_version=v1_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_base_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_base_${positive_file_name}_logs_seed_100.json
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

dataset=CLINC150
negate_prompt_version=v1
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_base_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/instructor_base_${positive_file_name}_logs_seed_100.json
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

# ----- IAE -----
dataset=BANKING77
negate_prompt_version=v4_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/iae_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/iae_${positive_file_name}_logs_seed_100.json
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

dataset=HWU64
negate_prompt_version=v1_2
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/iae_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/iae_${positive_file_name}_logs_seed_100.json
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

dataset=CLINC150
negate_prompt_version=v1
original_data_path=datasets/${dataset}/test
negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
save_emb_path=evaluate_results/${dataset}/triplet_task/implicature/iae_${positive_file_name}_embeds_seed_100.hdf5
save_log_path=evaluate_results/${dataset}/triplet_task/implicature/iae_${positive_file_name}_logs_seed_100.json
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