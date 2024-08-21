# export AWS_PROFILE=yuwzhan

model_name=falcon-40b-instruct
data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_parsed.json
prompt_path=build_toolkit_with_endpoints/prompts/summarize_intent_v5.txt
python build_toolkit_with_endpoints/summarize_intent.py \
    --prompt_path $prompt_path \
    --data_path $data_path \
    --model_name $model_name \
    --temperature 0. \
    --max_token 4 \
    --save_every 500
