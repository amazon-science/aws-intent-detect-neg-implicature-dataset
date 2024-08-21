export OPENAI_API_KEY="OPENAI_API_KEY"
export OPENAI_ORG_KEY="OPENAI_ORG_KEY"

# ===== v1 =====
# model_name="gpt-3.5-turbo-0613"
# challenge="negation"
# for dataset in BANKING77
# do
#     prompt_path="build_toolkit/prompts/negate_intent_${dataset}_v4_2.txt"
#     file_path="datasets/${dataset}/test"
#     original_label_path="datasets/${dataset}/explanations"
#     modified_label_path="datasets/${dataset}/negated_candidates"
#     python build_toolkit/modify_intent.py \
#         --dataset $dataset \
#         --prompt_path $prompt_path \
#         --file_path $file_path \
#         --original_label_path $original_label_path \
#         --modified_label_path $modified_label_path \
#         --model_name $model_name \
#         --challenge $challenge \
#         --temperature 0.
# done

# model_name="gpt-3.5-turbo-0613"
# challenge="negation"
# for dataset in HWU64
# do
#     prompt_path="build_toolkit/prompts/negate_intent_${dataset}_v1_2.txt"
#     file_path="datasets/${dataset}/test"
#     original_label_path="datasets/${dataset}/explanations"
#     modified_label_path="datasets/${dataset}/negated_candidates"
#     python build_toolkit/modify_intent.py \
#         --dataset $dataset \
#         --prompt_path $prompt_path \
#         --file_path $file_path \
#         --original_label_path $original_label_path \
#         --modified_label_path $modified_label_path \
#         --model_name $model_name \
#         --challenge $challenge \
#         --temperature 0. \
#         --overwrite
# done

# model_name="gpt-3.5-turbo-0613"
# challenge="negation"
# for dataset in CLINC150
# do
#     prompt_path="build_toolkit/prompts/negate_intent_${dataset}_v1.txt"
#     file_path="datasets/${dataset}/test"
#     original_label_path="datasets/${dataset}/explanations"
#     modified_label_path="datasets/${dataset}/negated_candidates"
#     python build_toolkit/modify_intent.py \
#         --dataset $dataset \
#         --prompt_path $prompt_path \
#         --file_path $file_path \
#         --original_label_path $original_label_path \
#         --modified_label_path $modified_label_path \
#         --model_name $model_name \
#         --challenge $challenge \
#         --temperature 0. \
#         --overwrite
# done

# ===== v2 test set =====
model_name="gpt-4-0613"
challenge="negation"
for dataset in BANKING77 HWU64 CLINC150
do
    prompt_path="build_toolkit/prompts/negate_intent_all_v2.txt"
    file_path="datasets/${dataset}/test"
    original_label_path="datasets/${dataset}/explanations"
    modified_label_path="datasets/${dataset}/negated_candidates_multi"
    python build_toolkit/modify_intent_v2.py \
        --dataset $dataset \
        --prompt_path $prompt_path \
        --file_path $file_path \
        --original_label_path $original_label_path \
        --modified_label_path $modified_label_path \
        --model_name $model_name \
        --challenge $challenge \
        --temperature 0.
done

# ===== v2 dev set =====
# this is only for prompt selection or any other hyper-parameter tuning
model_name="gpt-4-0613"
challenge="negation_devset"
for dataset in BANKING77 HWU64 CLINC150
do
    prompt_path="build_toolkit/prompts/negate_intent_all_v2.txt"
    file_path="datasets/${dataset}/dev"
    original_label_path="datasets/${dataset}/explanations"
    modified_label_path="datasets/${dataset}/negated_candidates_multi"
    python build_toolkit/modify_intent_v2.py \
        --dataset $dataset \
        --prompt_path $prompt_path \
        --file_path $file_path \
        --original_label_path $original_label_path \
        --modified_label_path $modified_label_path \
        --model_name $model_name \
        --challenge $challenge \
        --temperature 0. \
        --overwrite
done