#!/bin/bash -x

# for dataset in pilot_round
# do
#     python human_eval/read_results.py \
#         --dataset ${dataset}
# done

python human_eval/read_results_v3.py --dataset official_round_splitted_result