#!/bin/bash -x

# ===== BANKING77 =====
cd BANKING77
# will through an error if they exist, no need to comment them
git clone https://github.com/PolyAI-LDN/task-specific-datasets.git
mkdir test
# full/10-shot/5-shot
mkdir train
mkdir train_10
mkdir train_5
python convert_data.py
cd ../
# ===== HWU64 =====
cd HWU64
# will through an error if they exist, no need to comment them
git clone https://github.com/xliuhw/NLU-Evaluation-Data.git
mkdir test
# full/10-shot/5-shot
mkdir train
mkdir train_10
mkdir train_5
python convert_data.py
cd ../
# ===== CLINC150 =====
cd CLINC150
# will through an error if they exist, no need to comment them
git clone https://github.com/clinc/oos-eval.git
mkdir test
# full/10-shot/5-shot
mkdir train
mkdir train_10
mkdir train_5
python convert_data.py
cd ../
# ===== MultiWoz =====
cd MultiWOZ
# will through an error if they exist, no need to comment them
git clone https://github.com/budzianowski/multiwoz.git
mkdir train
python convert_data.py
cd ../
# ===== SGD =====
cd SGD
# will through an error if they exist, no need to comment them
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
mkdir train
python convert_data.py
cd ../
# ===== TOP =====
cd TOP
# download from http://fb.me/semanticparsingdialog on your browser and move it there
unzip -n semanticparsingdialog.zip
mkdir train
python convert_data.py
cd ../
# ===== TOPv2 =====
cd TOPv2
# download from https://fb.me/TOPv2Dataset on your browser and move it there
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -n TOPv2_Dataset.zip -d TOPv2
mkdir train
python convert_data.py
cd ../
# ===== DSTC11 =====
cd DSTC11-T2
# will through an error if they exist, no need to comment them
git clone https://github.com/amazon-science/dstc11-track2-intent-induction.git
mkdir train
python convert_data.py
cd ../

cd pretrain
python merge.py
cd ../