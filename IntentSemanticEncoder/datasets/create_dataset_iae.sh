#!/bin/bash -x

python create_pretrain_dataset_iae.py \
    --irl_model_path ../models/irl_model/irl-model-sgd-08-16-2022.tar.gz \
    --top1_dir TOP/top-dataset-semantic-parsing \
    --top2_dir TOPv2/TOPv2 \
    --dstc11t2_dir DSTC11-T2/dstc11-track2-intent-induction/dstc11 \
    --sgd_dir SGD/dstc8-schema-guided-dialogue \
    --multiwoz_dir MultiWOZ/multiwoz/data/MultiWOZ_2.2 \
    --output_dir iae_pretrain_data