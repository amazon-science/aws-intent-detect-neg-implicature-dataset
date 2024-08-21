data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3.json
# data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_partial.json
python build_toolkit_with_endpoints/parse_intents.py \
    --data_path $data_path
