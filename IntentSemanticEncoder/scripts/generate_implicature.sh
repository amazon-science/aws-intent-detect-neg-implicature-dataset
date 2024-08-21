export OPENAI_API_KEY="OPENAI_API_KEY"
export OPENAI_ORG_KEY="OPENAI_ORG_KEY"
# model_name="gpt-3.5-turbo-0613"
model_name="gpt-4-0613"
num=3
for dataset in BANKING77 HWU64 CLINC150
do
    # prompt_path="build_toolkit/prompts/generate_implicature_v1.txt"
    prompt_path="build_toolkit/prompts/in_context_implicature.txt"
    scenario_path="build_toolkit/results/${dataset}/implicature/scenarios_gpt-3.5-turbo-0613_brain_storm_v2.json"
    python build_toolkit/generate_implicature.py \
        --dataset $dataset \
        --prompt_path $prompt_path \
        --scenario_path $scenario_path \
        --model_name $model_name \
        --num $num \
        --temperature 0.
done