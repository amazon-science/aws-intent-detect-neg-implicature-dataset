#!/bin/bash -x

# split="dev"
split="test"

# ===== original =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path=datasets/${dataset}/${split}
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/original/instructor_large_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/original/instructor_large.json
    python evaluate_code/clustering.py \
        --test_data_path $test_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-large" \
        --overwrite
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path=datasets/${dataset}/${split}
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/original/instructor_base_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/original/instructor_base.json
    python evaluate_code/clustering.py \
        --test_data_path $test_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-base" \
        --overwrite
done

# ----- IAE -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path=datasets/${dataset}/${split}
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/original/iae_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/original/iae.json
    python evaluate_code/clustering.py \
        --test_data_path $test_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "iae" \
        --model_name_or_path "models/iae_model" \
        --overwrite
done

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path=datasets/${dataset}/${split}
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/original/paraphrase_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/original/paraphrase.json
    python evaluate_code/clustering.py \
        --test_data_path $test_data_path \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
        --overwrite
done

# ===== implicature =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    test_data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/implicature/instructor_large_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/implicature/instructor_large.json
    python evaluate_code/clustering.py \
        --test_data_path_seqin $test_data_path_seqin \
        --test_data_path_label $test_data_path_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-large" \
        --overwrite
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    test_data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/implicature/instructor_base_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/implicature/instructor_base.json
    python evaluate_code/clustering.py \
        --test_data_path_seqin $test_data_path_seqin \
        --test_data_path_label $test_data_path_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "hkunlp/instructor-base" \
        --overwrite
done

# ----- IAE -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    test_data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/implicature/iae_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/implicature/iae.json
    python evaluate_code/clustering.py \
        --test_data_path_seqin $test_data_path_seqin \
        --test_data_path_label $test_data_path_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "iae" \
        --model_name_or_path "models/iae_model" \
        --overwrite
done

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    test_data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
    test_data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
    save_emb_path=evaluate_results/${dataset}/clustering_${split}set/implicature/paraphrase_embeds.hdf5
    save_log_path=evaluate_results/${dataset}/clustering_${split}set/implicature/paraphrase.json
    python evaluate_code/clustering.py \
        --test_data_path_seqin $test_data_path_seqin \
        --test_data_path_label $test_data_path_label \
        --save_emb_path $save_emb_path \
        --save_log_path $save_log_path \
        --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
        --overwrite
done