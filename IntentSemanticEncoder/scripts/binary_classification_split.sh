#!/bin/bash -x

# split="dev"
split="test"

# ===== original =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path=datasets/${dataset}/${split}
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/original/instructor_large_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/original/instructor_large_logs.json
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
    data_path=datasets/${dataset}/${split}
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/original/instructor_base_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/original/instructor_base_logs.json
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
    data_path=datasets/${dataset}/${split}
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/original/iae_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/original/iae_logs.json
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

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path=datasets/${dataset}/${split}
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/original/paraphrase_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/original/paraphrase_logs.json
    python evaluate_code/binary_classification.py \
        --data_path $data_path \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
        --overwrite
done

# ===== implicature =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/instructor_large_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/instructor_large_logs.json
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
    data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/instructor_base_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/instructor_base_logs.json
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
    data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/iae_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/iae_logs.json
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

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/paraphrase_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/implicature/paraphrase_logs.json
    python evaluate_code/binary_classification.py \
        --data_path_seqin $data_path_seqin \
        --data_path_label $data_path_label \
        --candidate_path $candidate_path \
        --neg_candidate_path $neg_candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
        --overwrite
done

# ===== negated =====

# !warning: swap the negated and original candidates

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    data_path_label=datasets/${dataset}/${split}/label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/instructor_large_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/instructor_large_logs.json
    python evaluate_code/binary_classification.py \
        --data_path_seqin $data_path_seqin \
        --data_path_label $data_path_label \
        --candidate_path $neg_candidate_path \
        --neg_candidate_path $candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-large" \
        --overwrite
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    data_path_label=datasets/${dataset}/${split}/label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/instructor_base_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/instructor_base_logs.json
    python evaluate_code/binary_classification.py \
        --data_path_seqin $data_path_seqin \
        --data_path_label $data_path_label \
        --candidate_path $neg_candidate_path \
        --neg_candidate_path $candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-base" \
        --overwrite
done

# ----- iae -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    data_path_label=datasets/${dataset}/${split}/label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/iae_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/iae_logs.json
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
done

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    data_path_seqin=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
    data_path_label=datasets/${dataset}/${split}/label
    candidate_path=datasets/${dataset}/explanations
    neg_candidate_path=datasets/${dataset}/negated_candidates
    save_emb_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/paraphrase_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/binary_classification_${split}set/negated/paraphrase_logs.json
    python evaluate_code/binary_classification.py \
        --data_path_seqin $data_path_seqin \
        --data_path_label $data_path_label \
        --candidate_path $neg_candidate_path \
        --neg_candidate_path $candidate_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
        --overwrite
done
