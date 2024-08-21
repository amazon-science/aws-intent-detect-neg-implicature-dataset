#!/bin/bash -x

echo "!Deprecated, you should use \"binary_classification_v2.sh\""
exit 1

# ===== original =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path=datasets/${dataset}/test
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification/original/instructor_large_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification/original/instructor_large_logs.json
    python evaluate_code/binary_classification.py \
        --data_path $data_path \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-large" \
        --overwrite
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path=datasets/${dataset}/test
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification/original/instructor_base_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification/original/instructor_base_logs.json
    python evaluate_code/binary_classification.py \
        --data_path $data_path \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-base" \
        --overwrite
done

# ----- iae -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path=datasets/${dataset}/test
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification/original/iae_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification/original/iae_logs.json
    python evaluate_code/binary_classification.py \
        --data_path $data_path \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "iae" \
        --model_name_or_path models/iae_model \
        --overwrite
done

# ===== implicature =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
    data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification/implicature/instructor_large_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification/implicature/instructor_large_logs.json
    python evaluate_code/binary_classification.py \
        --data_path_seqin $data_path_seqin \
        --data_path_label $data_path_label \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-large" \
        --overwrite
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
    data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification/implicature/instructor_base_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification/implicature/instructor_base_logs.json
    python evaluate_code/binary_classification.py \
        --data_path_seqin $data_path_seqin \
        --data_path_label $data_path_label \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-base" \
        --overwrite
done

# ----- iae -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
    data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification/implicature/iae_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification/implicature/iae_logs.json
    python evaluate_code/binary_classification.py \
        --data_path_seqin $data_path_seqin \
        --data_path_label $data_path_label \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "iae" \
        --model_name_or_path models/iae_model \
        --overwrite
done

# ===== negated =====

# !warning: swap the negated and original candidates

# ----- instructor-large -----
dataset=BANKING77
negate_prompt_version=v4_2
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/instructor_large_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/instructor_large_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-large" \
    --overwrite

dataset=HWU64
negate_prompt_version=v1_2
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/instructor_large_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/instructor_large_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-large" \
    --overwrite

dataset=CLINC150
negate_prompt_version=v1
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/instructor_large_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/instructor_large_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-large" \
    --overwrite

# ----- instructor-base -----
dataset=BANKING77
negate_prompt_version=v4_2
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/instructor_base_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/instructor_base_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-base" \
    --overwrite

dataset=HWU64
negate_prompt_version=v1_2
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/instructor_base_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/instructor_base_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-base" \
    --overwrite

dataset=CLINC150
negate_prompt_version=v1
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/instructor_base_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/instructor_base_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "hkunlp/instructor-base" \
    --overwrite

# ----- iae -----
dataset=BANKING77
negate_prompt_version=v4_2
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/iae_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/iae_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "iae" \
    --model_name_or_path models/iae_model \
    --overwrite

dataset=HWU64
negate_prompt_version=v1_2
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/iae_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/iae_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "iae" \
    --model_name_or_path models/iae_model \
    --overwrite

dataset=CLINC150
negate_prompt_version=v1
data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
data_path_label=datasets/${dataset}/test/label
candidate_path=datasets/${dataset}/explanations
neg_candidate_path=datasets/${dataset}/negated_candidates
save_emb_path=evaluate_results/${dataset}/binary_classification/negated/iae_embeds.hdf5
save_log_path=evaluate_results/${dataset}/binary_classification/negated/iae_logs.json
python evaluate_code/binary_classification.py \
    --data_path_seqin $data_path_seqin \
    --data_path_label $data_path_label \
    --candidate_path $neg_candidate_path \
    --neg_candidate_path $candidate_path \
    --save_emb_path $save_emb_path \
    --save_log_path $save_log_path \
    --encoder_name "iae" \
    --model_name_or_path models/iae_model \
    --overwrite