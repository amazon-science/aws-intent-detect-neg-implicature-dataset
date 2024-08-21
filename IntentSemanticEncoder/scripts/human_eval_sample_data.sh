#!/bin/bash -x

# ===== BANKING77 =====
# orig_data_path=datasets/BANKING77/test
# imp_data_path_seqin=build_toolkit/results/BANKING77/implicature/utterances_gpt-4-0613_in_context_implicature.in
# imp_data_path_label=build_toolkit/results/BANKING77/implicature/utterances_gpt-4-0613_in_context_implicature_label
# neg_data_path_seqin=build_toolkit/results/BANKING77/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_BANKING77_v4_2.in
# neg_data_path_label=datasets/BANKING77/test/label
# candidate_path=datasets/BANKING77/explanations_for_human_eval
# python human_eval/sample_data.py \
#     --orig_data_path $orig_data_path \
#     --imp_data_path_seqin $imp_data_path_seqin \
#     --imp_data_path_label $imp_data_path_label \
#     --neg_data_path_seqin $neg_data_path_seqin \
#     --neg_data_path_label $neg_data_path_label \
#     --candidate_path $candidate_path \
#     --dataset BANKING77

# ===== HWU64 =====
# orig_data_path=datasets/HWU64/test
# imp_data_path_seqin=build_toolkit/results/HWU64/implicature/utterances_gpt-4-0613_in_context_implicature.in
# imp_data_path_label=build_toolkit/results/HWU64/implicature/utterances_gpt-4-0613_in_context_implicature_label
# neg_data_path_seqin=build_toolkit/results/HWU64/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_HWU64_v1_2.in
# neg_data_path_label=datasets/HWU64/test/label
# candidate_path=datasets/HWU64/explanations
# python human_eval/sample_data.py \
#     --orig_data_path $orig_data_path \
#     --imp_data_path_seqin $imp_data_path_seqin \
#     --imp_data_path_label $imp_data_path_label \
#     --neg_data_path_seqin $neg_data_path_seqin \
#     --neg_data_path_label $neg_data_path_label \
#     --candidate_path $candidate_path \
#     --dataset HWU64

# ===== CLINC150 =====
# orig_data_path=datasets/CLINC150/test
# imp_data_path_seqin=build_toolkit/results/CLINC150/implicature/utterances_gpt-4-0613_in_context_implicature.in
# imp_data_path_label=build_toolkit/results/CLINC150/implicature/utterances_gpt-4-0613_in_context_implicature_label
# neg_data_path_seqin=build_toolkit/results/CLINC150/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_CLINC150_v1.in
# neg_data_path_label=datasets/CLINC150/test/label
# candidate_path=datasets/CLINC150/explanations
# python human_eval/sample_data.py \
#     --orig_data_path $orig_data_path \
#     --imp_data_path_seqin $imp_data_path_seqin \
#     --imp_data_path_label $imp_data_path_label \
#     --neg_data_path_seqin $neg_data_path_seqin \
#     --neg_data_path_label $neg_data_path_label \
#     --candidate_path $candidate_path \
#     --dataset CLINC150

# ===== v2 =====
# python human_eval/sample_data_v2.py --seed 1234
# python human_eval/sample_data_v2.py --seed 888 --budget 5 --dataset_name pilot_round_2
# python human_eval/sample_data_v2.py --seed 4321 --budget 5 --dataset_name pilot_round_3
# python human_eval/sample_data_v2.py --seed 666 --budget 20 --dataset_name official_round
python human_eval/sample_data_v3.py --seed 666 --budget 20 --dataset_name official_round_splitted