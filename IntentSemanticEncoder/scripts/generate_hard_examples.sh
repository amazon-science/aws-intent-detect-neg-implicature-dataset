# export AWS_PROFILE=yuwzhan

model_name=falcon-40b-instruct
data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5.json
python build_toolkit_with_endpoints/generate_hard_examples.py \
    --data_path $data_path \
    --model_name $model_name \
    --temperature 0. \
    --max_token 256 \
    --save_every 200
