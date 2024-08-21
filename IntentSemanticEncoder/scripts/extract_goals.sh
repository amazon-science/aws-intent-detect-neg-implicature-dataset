# You need to export credentials from isengard for running the following script.
# export AWS_PROFILE=yuwzhan


# model_name=mpt-30b-chat
model_name=falcon-40b-instruct
data_path=datasets/pretrain/seq.in
prompt_path=build_toolkit_with_endpoints/prompts/extract_intent_v3.txt
python build_toolkit_with_endpoints/extract_goals.py \
    --prompt_path $prompt_path \
    --data_path $data_path \
    --model_name $model_name \
    --temperature 0. \
    --save_every 200
