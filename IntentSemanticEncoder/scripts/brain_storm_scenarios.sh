export OPENAI_API_KEY="OPENAI_API_KEY"
export OPENAI_ORG_KEY="OPENAI_ORG_KEY"
model_name="gpt-3.5-turbo-0613"
num=10
for dataset in BANKING77 HWU64 CLINC150
do
    prompt_path="build_toolkit/prompts/brain_storm_v2.txt"
    original_label_path="datasets/${dataset}/explanations"
    domain_path="datasets/${dataset}/domains"
    python build_toolkit/brain_storm_scenarios.py \
        --dataset $dataset \
        --prompt_path $prompt_path \
        --original_label_path $original_label_path \
        --domain_path $domain_path \
        --model_name $model_name \
        --num $num \
        --temperature 0. \
        --max_token 1280 \
        --overwrite
done