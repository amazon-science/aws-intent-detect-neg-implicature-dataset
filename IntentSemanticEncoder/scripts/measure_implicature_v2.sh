#!/bin/bash -x

# !warning: top_k argument will not affect methods other than 'keyword'

# ===== original =====
for dataset in BANKING77 HWU64 CLINC150
do
    for method in keyword BLEU ROUGE METEOR BertScore
    do
        train_data_path=datasets/${dataset}/train
        test_data_path=datasets/${dataset}/test
        save_log_path=evaluate_results/${dataset}/measure_implicature/original/${method}_logs.json
        python build_toolkit/measure_implicature.py \
            --train_data_path $train_data_path \
            --test_data_path $test_data_path \
            --save_log_path $save_log_path \
            --method $method \
            --top_k 15
    done
done


# ===== implicature gpt-4 =====
for dataset in BANKING77 HWU64 CLINC150
do
    for method in keyword BLEU ROUGE METEOR BertScore
    do
        train_data_path=datasets/${dataset}/train
        test_data_path_seqin=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in
        test_data_path_label=build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label
        save_log_path=evaluate_results/${dataset}/measure_implicature/implicature/${method}_logs.json
        python build_toolkit/measure_implicature.py \
            --train_data_path $train_data_path \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --save_log_path $save_log_path \
            --method $method \
            --top_k 15
    done
done

# ===== negation =====
for dataset in BANKING77 HWU64 CLINC150
do
    for method in keyword BLEU ROUGE METEOR BertScore
    do
        train_data_path=datasets/${dataset}/train
        test_data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-4-0613_negate_intent_all_v2.in
        test_data_path_label=datasets/${dataset}/test/label
        save_log_path=evaluate_results/${dataset}/measure_implicature/negation/${method}_logs.json
        python build_toolkit/measure_implicature.py \
            --train_data_path $train_data_path \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --save_log_path $save_log_path \
            --method $method \
            --top_k 15
    done
done
# ===== implicature mpt-30b-chat =====

# for dataset in BANKING77
# do
#     for method in keyword BLEU ROUGE METEOR BertScore
#     do
#         train_data_path=datasets/${dataset}/train
#         test_data_path_seqin=build_toolkit_with_endpoints/results/${dataset}/implicature/utterance_mpt-30b-chat_in_context_implicature.in
#         test_data_path_label=build_toolkit_with_endpoints/results/${dataset}/implicature/utterance_mpt-30b-chat_in_context_implicature_label
#         save_log_path=evaluate_results/${dataset}/measure_implicature/implicature_mpt-30b-chat/${method}_logs.json
#         python build_toolkit/measure_implicature.py \
#             --train_data_path $train_data_path \
#             --test_data_path_seqin $test_data_path_seqin \
#             --test_data_path_label $test_data_path_label \
#             --save_log_path $save_log_path \
#             --method $method \
#             --top_k 15
#     done
# done
