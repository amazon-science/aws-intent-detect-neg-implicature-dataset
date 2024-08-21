#!/bin/bash -x

# uncomment one of the code block below and add arguments you want to start training.

export CUDA_VISIBLE_DEVICES=0

# ===== pilot round (25% data) =====

# ----- instructor-base -----
# output_directory=pretrain/results/pretrain_data_v1_55397_proc_disable_hard_pos_6
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v1_55397_proc.json
# python pretrain_code/train.py \
#     --model_name_or_path hkunlp/instructor-base \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 5 \
#     --save_strategy 'epoch' \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 2e-5 \
#     --run_name pretrain_data_v1_55397_proc_disable_hard_pos_6 \
#     --disable_hard_positive_types "hard_positive_6.txt" \
#     --overwrite_output_dir

# ----- instructor-large -----
# output_directory=pretrain/results/pretrain_data_v2_69395_proc_disable_hard_pos_7_hard_neg_4
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v2_69395_proc.json
# python pretrain_code/train.py \
#     --model_name_or_path hkunlp/instructor-large \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 5 \
#     --save_strategy 'epoch' \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 2e-5 \
#     --run_name pretrain_data_v2_69395_proc_disable_hard_pos_7_hard_neg_4 \
#     --disable_hard_negative_types "hard_negative_4.txt" \
#     --disable_hard_positive_types "hard_positive_7.txt" \
#     --overwrite_output_dir

# ----- paraphrase -----
# output_directory=pretrain/results/pretrain_data_v5_paraphrase_97989_proc_disable_hard_pos_7_4
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v5_paraphrase_97989_proc.json
# python pretrain_code/train_st.py \
#     --model_name_or_path sentence-transformers/paraphrase-mpnet-base-v2 \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 1 \
#     --save_strategy 'steps' \
#     --save_steps 7000 \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 1e-6 \
#     --run_name pretrain_data_v5_paraphrase_97989_proc_disable_hard_pos_7_4 \
#     --disable_hard_positive_types "hard_positive_7.txt,hard_positive_4.txt" \
#     --overwrite_output_dir

# ----- iae -----
# output_directory=pretrain/results/pretrain_data_v6_iae_54997_proc_disable_hard_pos_7_hard_neg_2
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v6_iae_54997_proc.json
# python pretrain_code/train_st.py \
#     --model_name_or_path models/iae_model \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 1 \
#     --save_strategy 'steps' \
#     --save_steps 7000 \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 1e-6 \
#     --run_name pretrain_data_v6_iae_54997_proc_disable_hard_pos_7_hard_neg_2 \
#     --disable_hard_positive_types "hard_positive_7.txt" \
#     --disable_hard_negative_types "hard_negative_2.txt" \
#     --overwrite_output_dir

# ===== official round (100% data) =====

# ----- paraphrase -----
# output_directory=pretrain/results/pretrain_data_v7_paraphrase_252744_proc_disable_llm
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v7_paraphrase_252744_proc.json
# python pretrain_code/train_st.py \
#     --model_name_or_path sentence-transformers/paraphrase-mpnet-base-v2 \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 1 \
#     --save_strategy 'steps' \
#     --save_steps 7000 \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 1e-6 \
#     --run_name pretrain_data_v7_paraphrase_252744_proc_disable_llm \
#     --disable_llm_generated \
#     --overwrite_output_dir

# ----- iae -----
# output_directory=pretrain/results/pretrain_data_v8_iae_252744_proc_disable_hard_pos_7_hard_neg_2_4_names
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v8_iae_252744_proc.json
# python pretrain_code/train_st.py \
#     --model_name_or_path models/iae_model \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 1 \
#     --save_strategy 'steps' \
#     --save_steps 7000 \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 1e-6 \
#     --run_name pretrain_data_v8_iae_252744_proc_disable_hard_pos_7_hard_neg_2_4_names \
#     --disable_hard_positive_types "hard_positive_7.txt" \
#     --disable_hard_negative_types "hard_negative_2.txt,hard_negative_4.txt" \
#     --add_positive_intent_names \
#     --add_negative_intent_names \
#     --overwrite_output_dir

# ----- instructor-base -----
# output_directory=pretrain/results/pretrain_data_v3_instructor_base_252744_proc_disable_llm
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v3_instructor_base_252744_proc.json
# python pretrain_code/train.py \
#     --model_name_or_path hkunlp/instructor-base \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 1 \
#     --save_strategy 'steps' \
#     --save_steps 7000 \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 4e-6 \
#     --run_name pretrain_data_v3_instructor_base_252744_proc_disable_llm \
#     --disable_llm_generated \
#     --overwrite_output_dir

# ----- instructor-large -----
# output_directory=pretrain/results/pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_hard_neg_4_lr
# cache_directory=cache
# train_file=build_toolkit_with_endpoints/results/pretrain/pretrain_data_v4_instructor_large_252744_proc.json
# python pretrain_code/train.py \
#     --model_name_or_path hkunlp/instructor-large \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 1 \
#     --save_strategy 'steps' \
#     --save_steps 7000 \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 4e-6 \
#     --run_name pretrain_data_v4_instructor_large_252744_proc_disable_hard_pos_7_hard_neg_4_lr \
#     --disable_hard_positive_types "hard_positive_7.txt" \
#     --disable_hard_negative_types "hard_negative_4.txt" \
#     --overwrite_output_dir

# ===== paraphrase model comparative study =====
# add 
# default seed is 42
# output_directory=pretrain/results/pretrain_data_paraphrase_comparative_seed666
# cache_directory=cache
# train_file=datasets/iae_pretrain_data_llmaug_v2/train.json
# python pretrain_code/train_st_v2.py \
#     --model_name_or_path sentence-transformers/paraphrase-mpnet-base-v2 \
#     --train_file ${train_file} \
#     --output_dir ${output_directory} \
#     --cache_dir ${cache_directory} \
#     --max_source_length 128 \
#     --num_train_epochs 1 \
#     --save_strategy 'steps' \
#     --save_steps 7000 \
#     --cl_temperature 0.01 \
#     --warmup_ratio 0.1 \
#     --learning_rate 1e-6 \
#     --run_name pretrain_data_paraphrase_comparative_seed666 \
#     --overwrite_output_dir \
#     --seed 666
