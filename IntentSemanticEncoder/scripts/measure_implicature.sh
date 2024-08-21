#!/bin/bash -x

echo "!Deprecated, you should use \"measure_implicature_v2.sh\""
exit 1

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
dataset=BANKING77
negate_prompt_version=v4_2
for method in keyword BLEU ROUGE METEOR BertScore
do
    train_data_path=datasets/${dataset}/train
    test_data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
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

dataset=HWU64
negate_prompt_version=v1_2
for method in keyword BLEU ROUGE METEOR BertScore
do
    train_data_path=datasets/${dataset}/train
    test_data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
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

dataset=CLINC150
negate_prompt_version=v1
for method in keyword BLEU ROUGE METEOR BertScore
do
    train_data_path=datasets/${dataset}/train
    test_data_path_seqin=build_toolkit/results/${dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_${dataset}_${negate_prompt_version}.in
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

# ===== implicature mpt-30b-chat =====

for dataset in BANKING77
do
    for method in keyword BLEU ROUGE METEOR BertScore
    do
        train_data_path=datasets/${dataset}/train
        test_data_path_seqin=build_toolkit_with_endpoints/results/${dataset}/implicature/utterance_mpt-30b-chat_in_context_implicature.in
        test_data_path_label=build_toolkit_with_endpoints/results/${dataset}/implicature/utterance_mpt-30b-chat_in_context_implicature_label
        save_log_path=evaluate_results/${dataset}/measure_implicature/implicature_mpt-30b-chat/${method}_logs.json
        python build_toolkit/measure_implicature.py \
            --train_data_path $train_data_path \
            --test_data_path_seqin $test_data_path_seqin \
            --test_data_path_label $test_data_path_label \
            --save_log_path $save_log_path \
            --method $method \
            --top_k 15
    done
done
