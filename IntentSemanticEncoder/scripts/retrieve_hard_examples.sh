export CUDA_VISIBLE_DEVICES=0

data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json
python build_toolkit_with_endpoints/retrieve_hard_examples.py \
    --data_path $data_path \
    --encoder_name "hkunlp/instructor-large" \
    --overwrite

data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json
python build_toolkit_with_endpoints/retrieve_hard_examples.py \
    --data_path $data_path \
    --encoder_name "hkunlp/instructor-base" \
    --overwrite

data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json
python build_toolkit_with_endpoints/retrieve_hard_examples.py \
    --data_path $data_path \
    --encoder_name "sentence-transformers/paraphrase-mpnet-base-v2" \
    --save_every 1000 \
    --overwrite

# continue to train on IAE does not seem to work,
# because IAE uses the gold label but not ours.
# However, I showed in another experiment that adding LLM-generated data
# into IAE training framework can still help. Therefore the below code is
# not optimal for now. Deprecated!

# data_path=build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json
# python build_toolkit_with_endpoints/retrieve_hard_examples.py \
#     --data_path $data_path \
#     --encoder_name "models/iae_model" \
#     --save_every 1000 \
#     --overwrite
