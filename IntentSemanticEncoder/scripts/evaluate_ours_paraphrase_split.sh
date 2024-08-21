#!/bin/bash -x

# split="dev"
split="test"

# https://stackoverflow.com/a/8880633
declare -a model_list=(
    # "/fsx/users/yuwzhan/baseline/intent-aware-encoder/models/iae_model_llm"
    # "/fsx/users/yuwzhan/baseline/intent-aware-encoder/models/iae_model_llm_33"
    # "/fsx/users/yuwzhan/baseline/intent-aware-encoder/models/iae_model_llm_666"
    # "/fsx/users/yuwzhan/baseline/intent-aware-encoder/models/iae_model_normal"
    # "/fsx/users/yuwzhan/baseline/intent-aware-encoder/models/iae_model_normal_33"
    # "/fsx/users/yuwzhan/baseline/intent-aware-encoder/models/iae_model_normal_666"
    # "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_paraphrase_comparative"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_paraphrase_comparative_seed33"
    "/fsx/users/yuwzhan/codes/IntentSemanticEncoder/pretrain/results/pretrain_data_paraphrase_comparative_seed666"
)

for model_name in "${model_list[@]}"
do
    echo "evaluating ${model_name} ..."

    # ===== triplet task =====

    # ----- negation -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        original_data_path=datasets/${dataset}/${split}
        negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
        save_emb_path=${model_name}/${dataset}/triplet_task_${split}set/negation/embeds_seed_100.hdf5
        save_log_path=${model_name}/${dataset}/triplet_task_${split}set/negation/logs_seed_100.json
        python evaluate_code/triplet_task.py \
            --original_data_path $original_data_path \
            --negative_data_path $negative_data_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- implicature -----
    positive_file_name=utterances_gpt-4-0613_in_context_implicature

    for dataset in BANKING77 HWU64 CLINC150
    do
        original_data_path=datasets/${dataset}/${split}
        negative_data_path=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
        positive_data_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}.in
        positive_data_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/${positive_file_name}_label
        save_emb_path=${model_name}/${dataset}/triplet_task_${split}set/implicature/${positive_file_name}_embeds_seed_100.hdf5
        save_log_path=${model_name}/${dataset}/triplet_task_${split}set/implicature/${positive_file_name}_logs_seed_100.json
        python evaluate_code/triplet_task.py \
            --original_data_path $original_data_path \
            --negative_data_path $negative_data_path \
            --positive_data_seqin $positive_data_seqin \
            --positive_data_label $positive_data_label \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --model_name_or_path $model_name \
            --sample_positive \
            --overwrite
    done

    # ===== binary classification =====

    # ----- original -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        data_path=datasets/${dataset}/${split}
        candidate_path=datasets/${dataset}/explanations
        neg_candidate_path=datasets/${dataset}/negated_candidates
        save_emb_path=${model_name}/${dataset}/binary_classification_${split}set/original/embeds.hdf5
        save_log_path=${model_name}/${dataset}/binary_classification_${split}set/original/logs.json
        python evaluate_code/binary_classification.py \
            --data_path $data_path \
            --candidate_path $candidate_path \
            --neg_candidate_path $neg_candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- implicature -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
        data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
        candidate_path=datasets/${dataset}/explanations
        neg_candidate_path=datasets/${dataset}/negated_candidates
        save_emb_path=${model_name}/${dataset}/binary_classification_${split}set/implicature/embeds.hdf5
        save_log_path=${model_name}/${dataset}/binary_classification_${split}set/implicature/logs.json
        python evaluate_code/binary_classification.py \
            --data_path_seqin $data_path_seqin \
            --data_path_label $data_path_label \
            --candidate_path $candidate_path \
            --neg_candidate_path $neg_candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- negated -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        data_path_seqin=build_toolkit/results/${dataset}/negation_${split}set/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
        data_path_label=datasets/${dataset}/${split}/label
        candidate_path=datasets/${dataset}/explanations
        neg_candidate_path=datasets/${dataset}/negated_candidates
        save_emb_path=${model_name}/${dataset}/binary_classification_${split}set/negated/embeds.hdf5
        save_log_path=${model_name}/${dataset}/binary_classification_${split}set/negated/logs.json
        python evaluate_code/binary_classification.py \
            --data_path_seqin $data_path_seqin \
            --data_path_label $data_path_label \
            --candidate_path $neg_candidate_path \
            --neg_candidate_path $candidate_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ===== clustering =====

    # ----- original -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        test_data_path=datasets/${dataset}/${split}
        save_emb_path=${model_name}/${dataset}/clustering_${split}set/original/embeds.hdf5
        save_log_path=${model_name}/${dataset}/clustering_${split}set/original/logs.json
        python evaluate_code/clustering.py \
            --test_data_path $test_data_path \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
            --model_name_or_path $model_name \
            --overwrite
    done

    # ----- implicature -----
    for dataset in BANKING77 HWU64 CLINC150
    do
        test_data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
        test_data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
        save_emb_path=${model_name}/${dataset}/clustering_${split}set/implicature/embeds.hdf5
        save_log_path=${model_name}/${dataset}/clustering_${split}set/implicature/logs.json
        python evaluate_code/clustering.py \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --save_emb_path $save_emb_path \
            --save_log_path $save_log_path \
            --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
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
            test_data_path=datasets/${dataset}/${split}
            candidate_path=datasets/${dataset}/explanations
            save_emb_path=${model_name}/${dataset}/protonet_fullwayclass_${split}set/original/${num_shots}shots_embeds.hdf5
            save_log_path=${model_name}/${dataset}/protonet_fullwayclass_${split}set/original/${num_shots}shots_logs.json
            python evaluate_code/protonet.py \
                --train_data_path $train_data_path \
                --test_data_path $test_data_path \
                --candidate_path $candidate_path \
                --save_emb_path $save_emb_path \
                --save_log_path $save_log_path \
                --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
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
            test_data_path_seqin=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature.in
            test_data_path_label=build_toolkit/results/${dataset}/implicature_splitted/${split}/utterances_gpt-4-0613_in_context_implicature_label
            candidate_path=datasets/${dataset}/explanations
            save_emb_path=${model_name}/${dataset}/protonet_fullwayclass_${split}set/implicature/${num_shots}shots_embeds.hdf5
            save_log_path=${model_name}/${dataset}/protonet_fullwayclass_${split}set/implicature/${num_shots}shots_logs.json
            python evaluate_code/protonet.py \
                --train_data_path $train_data_path \
                --test_data_path_seqin $test_data_path_seqin \
                --test_data_path_label $test_data_path_label \
                --candidate_path $candidate_path \
                --save_emb_path $save_emb_path \
                --save_log_path $save_log_path \
                --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
                --model_name_or_path $model_name \
                --num_shots $num_shots \
                --overwrite
        done
    done

    # ----- average results -----
    cd pretrain
    python average_results_split.py --model_name $model_name --split $split
    cd ..
done