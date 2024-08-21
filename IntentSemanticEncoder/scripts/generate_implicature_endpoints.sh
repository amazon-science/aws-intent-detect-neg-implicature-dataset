export ISENGARD_PRODUCTION_ACCOUNT=false
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_SESSION_TOKEN=

model_name=mpt-30b-chat
num=3
for dataset in BANKING77
do
    prompt_path=build_toolkit_with_endpoints/prompts/in_context_implicature.txt
    scenario_path="build_toolkit/results/${dataset}/implicature/scenarios_gpt-3.5-turbo-0613_brain_storm_v2.json"
    python build_toolkit_with_endpoints/generate_implicature.py \
        --prompt_path $prompt_path \
        --scenario_path $scenario_path \
        --model_name $model_name \
        --num $num \
        --temperature 0. \
        --overwrite
done
