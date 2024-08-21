#!/bin/bash -x

# https://stackoverflow.com/a/8880633
# copy the absolute path to the model to the below list
# make sure there exists at least the `pytorch_model.bin`
# if there is a config.json then it will also load config.
# the results will be saved in the model folder.
# The code will also automatically save the averaged results across 3 datasets.
declare -a model_list=(
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_llm"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_positive_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_negative_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_4_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_5_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_6_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_hard_neg_2_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_hard_neg_3_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_hard_neg_4_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_hard_neg_2_4_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_4_hard_neg_2_4_lr"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_4_5_hard_neg_2_4_lr"
)

for model_name in "${model_list[@]}"
do
    echo "evaluating ${model_name} ..."

    # ===== triplet task =====

    # ----- negation -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        original_data_path=datasets/${dataset}/test
        negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
        save_emb_path=${model_name}/${dataset}/triplet_task/negation/embeds_seed_100.hdf5
        save_log_path=${model_name}/${dataset}/triplet_task/negation/logs_seed_100.json
        python evaluate_code/triplet_task.py \
            --original_data_path $original_data_path \
            --negative_data_path $negative_data_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- implicature -----
    positive_file_name=utterances_gpt-4-0613_in_context_implicature

    for dataset in BANKING77 HWU64 CLINC150
    do
        original_data_path=datasets/${dataset}/test
        negative_data_path=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
        positive_data_seqin=build_toolkit/results/${dataset}/implicature/${positive_file_name}.in
        positive_data_label=build_toolkit/results/${dataset}/implicature/${positive_file_name}_label
        save_emb_path=${model_name}/${dataset}/triplet_task/implicature/${positive_file_name}_embeds_seed_100.hdf5
        save_log_path=${model_name}/${dataset}/triplet_task/implicature/${positive_file_name}_logs_seed_100.json
        python evaluate_code/triplet_task.py \
            --original_data_path $original_data_path \
            --negative_data_path $negative_data_path \
            --positive_data_seqin $positive_data_seqin \
            --positive_data_label $positive_data_label \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --model_name_or_path $model_name \
            --sample_positive \
            --overwrite
    done

    # ===== binary classification =====

    # ----- original -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        data_path=datasets/${dataset}/test
        candidate_path=datasets/${dataset}/explanations
        neg_candidate_path=datasets/${dataset}/negated_candidates
        save_emb_path=${model_name}/${dataset}/binary_classification/original/embeds.hdf5
        save_log_path=${model_name}/${dataset}/binary_classification/original/logs.json
        python evaluate_code/binary_classification.py \
            --data_path $data_path \
            --candidate_path $candidate_path \
            --neg_candidate_path $neg_candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- implicature -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
        data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
        candidate_path=datasets/${dataset}/explanations
        neg_candidate_path=datasets/${dataset}/negated_candidates
        save_emb_path=${model_name}/${dataset}/binary_classification/implicature/embeds.hdf5
        save_log_path=${model_name}/${dataset}/binary_classification/implicature/logs.json
        python evaluate_code/binary_classification.py \
            --data_path_seqin $data_path_seqin \
            --data_path_label $data_path_label \
            --candidate_path $candidate_path \
            --neg_candidate_path $neg_candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- negated -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
        data_path_label=datasets/${dataset}/test/label
        candidate_path=datasets/${dataset}/explanations
        neg_candidate_path=datasets/${dataset}/negated_candidates
        save_emb_path=${model_name}/${dataset}/binary_classification/negated/embeds.hdf5
        save_log_path=${model_name}/${dataset}/binary_classification/negated/logs.json
        python evaluate_code/binary_classification.py \
            --data_path_seqin $data_path_seqin \
            --data_path_label $data_path_label \
            --candidate_path $neg_candidate_path \
            --neg_candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ===== clustering =====

    # ----- original -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        test_data_path=datasets/${dataset}/test
        save_emb_path=${model_name}/${dataset}/clustering/original/embeds.hdf5
        save_log_path=${model_name}/${dataset}/clustering/original/logs.json
        python evaluate_code/clustering.py \
            --test_data_path $test_data_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- implicature -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        test_data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
        test_data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
        save_emb_path=${model_name}/${dataset}/clustering/implicature/embeds.hdf5
        save_log_path=${model_name}/${dataset}/clustering/implicature/logs.json
        python evaluate_code/clustering.py \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "hkunlp/instructor-base" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ===== multi-class =====

    # ----- original -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        for num_shots in 0 10
        do
            train_data_path=datasets/${dataset}/train_${num_shots}
            test_data_path=datasets/${dataset}/test
            candidate_path=datasets/${dataset}/explanations
            save_emb_path=${model_name}/${dataset}/protonet_fullwayclass/original/${num_shots}shots_embeds.hdf5
            save_log_path=${model_name}/${dataset}/protonet_fullwayclass/original/${num_shots}shots_logs.json
            python evaluate_code/protonet.py \
                --train_data_path $train_data_path \
                --test_data_path $test_data_path \
                --candidate_path $candidate_path \
                --save_emb_path $save_emb_path \
                --save_log_path $save_log_path \
                --encoder_name "hkunlp/instructor-base" \
                --model_name_or_path $model_name \
                --num_shots $num_shots \
                --overwrite
        done
    done

    # ----- implicature -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        for num_shots in 0 10
        do
            train_data_path=datasets/${dataset}/train_${num_shots}
            test_data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
            test_data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
            candidate_path=datasets/${dataset}/explanations
            save_emb_path=${model_name}/${dataset}/protonet_fullwayclass/implicature/${num_shots}shots_embeds.hdf5
            save_log_path=${model_name}/${dataset}/protonet_fullwayclass/implicature/${num_shots}shots_logs.json
            python evaluate_code/protonet.py \
                --train_data_path $train_data_path \
                --test_data_path_seqin $test_data_path_seqin \
                --test_data_path_label $test_data_path_label \
                --candidate_path $candidate_path \
                --save_emb_path $save_emb_path \
                --save_log_path $save_log_path \
                --encoder_name "hkunlp/instructor-base" \
                --model_name_or_path $model_name \
                --num_shots $num_shots \
                --overwrite
        done
    done

    # ----- average results -----
    cd pretrain
    python average_results.py --model_name $model_name
    cd ..
done