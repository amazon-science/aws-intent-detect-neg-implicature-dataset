#!/bin/bash -x

# !warning: the script will run zero-shot with num_shots=0, even if you give it a train_data_path
# tips: when using instructor-large, without prompt is even better than with prompt on BANKING77, but I choose to add prompt because it makes sense to indicate the aspect we are encoding.
# tips: using explanation is better than using original label name

# ===== original =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path=datasets/${dataset}/test
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/original/instructor_large_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/original/instructor_large_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path $test_data_path \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-large" \
            --num_shots $num_shots \
            --overwrite
    done
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path=datasets/${dataset}/test
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/original/instructor_base_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/original/instructor_base_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path $test_data_path \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --num_shots $num_shots \
            --overwrite
    done
done

# ----- iae -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path=datasets/${dataset}/test
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/original/iae_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/original/iae_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path $test_data_path \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "iae" \
            --model_name_or_path models/iae_model \
            --num_shots $num_shots \
            --overwrite
    done
done

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path=datasets/${dataset}/test
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/original/paraphrase_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/original/paraphrase_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path $test_data_path \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --num_shots $num_shots \
            --overwrite
    done
done

# ===== implicature =====

# ----- instructor-large -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
        test_data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/instructor_large_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/instructor_large_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-large" \
            --num_shots $num_shots \
            --overwrite
    done
done

# ----- instructor-base -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
        test_data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/instructor_base_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/instructor_base_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --num_shots $num_shots \
            --overwrite
    done
done

# ----- iae -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
        test_data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/iae_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/iae_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "iae" \
            --model_name_or_path models/iae_model \
            --num_shots $num_shots \
            --overwrite
    done
done

# ----- paraphrase -----
for dataset in BANKING77 HWU64 CLINC150
do
    for num_shots in 0 10
    do
        train_data_path=datasets/${dataset}/train_${num_shots}
        test_data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
        test_data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
        candidate_path=datasets/${dataset}/explanations
        save_emb_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/paraphrase_${num_shots}shots_embeds.hdf5
        save_log_path=evaluate_results/${dataset}/protonet_fullwayclass/implicature/paraphrase_${num_shots}shots_logs.json
        python evaluate_code/protonet.py \
            --train_data_path $train_data_path \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --num_shots $num_shots \
            --overwrite
    done
done