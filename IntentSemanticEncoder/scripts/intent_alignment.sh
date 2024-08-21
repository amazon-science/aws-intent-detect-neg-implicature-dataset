export OPENAI_API_KEY="OPENAI_API_KEY"
export OPENAI_ORG_KEY="OPENAI_ORG_KEY"
model_name="gpt-4-0613"
topk=5
prompt_path="build_toolkit/prompts/intent_alignment_v1.txt"

for dataset in BANKING77
do
    candidate_path="datasets/${dataset}/explanations"
    test_data_path_seqin="build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in"
    test_data_path_label="build_toolkit/results/${dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label"
    log_path="evaluate_results/${dataset}/protonet_fullwayclass/implicature/instructor_large_10shots_logs.json"
    python build_toolkit/intent_alignment.py \
        --dataset $dataset \
        --prompt_path $prompt_path \
        --test_data_path_seqin $test_data_path_seqin \
        --test_data_path_label $test_data_path_label \
        --candidate_path $candidate_path \
        --log_path $log_path \
        --model_name $model_name \
        --overwrite
done