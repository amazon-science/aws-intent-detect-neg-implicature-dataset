import os, csv

data_dir = "NLU-Evaluation-Data/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation/KFold_1"

assert os.path.exists(data_dir)

# ===== test =====

test_dir = os.path.join(data_dir, "testset/text")

assert os.path.exists(test_dir)

texts, labels = [], []
for file_name in sorted(os.listdir(test_dir)):
    label_name = os.path.splitext(file_name)[0]
    if label_name == "general_quirky":
        continue
    with open(os.path.join(test_dir, file_name), 'r') as f:
        new_texts = [l.strip() for l in f.readlines()]
        texts.extend(new_texts)
    labels.extend([label_name] * len(new_texts))

output_dir = "test"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, 'seq.in'), 'w', encoding='utf-8') as f_text, \
    open(os.path.join(output_dir, 'label'), 'w', encoding='utf-8') as f_label:
    f_text.writelines([t.strip().lower() + "\n" for t in texts])
    f_label.writelines([l + "\n" for l in labels])

# ===== train =====

train_dir = os.path.join(data_dir, "trainset")

assert os.path.exists(train_dir)

texts, labels = [], []
for file_name in sorted(os.listdir(train_dir)):
    label_name = os.path.splitext(file_name)[0]
    if label_name == "general_quirky":
        continue
    with open(os.path.join(train_dir, file_name), 'r') as f:
        csv_reader = csv.reader(f, delimiter=";")
        new_texts = [l[4].strip().lower() for l in csv_reader if l[0] != "answerid"]
        texts.extend(new_texts)
    labels.extend([label_name] * len(new_texts))

output_dir = "train"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, 'seq.in'), 'w', encoding='utf-8') as f_text, \
    open(os.path.join(output_dir, 'label'), 'w', encoding='utf-8') as f_label:
    f_text.writelines([t.strip().lower() + "\n" for t in texts])
    f_label.writelines([l + "\n" for l in labels])

# ===== train_10 =====

import random
from collections import defaultdict
random.seed(0)

label2data = defaultdict(list)
for t, l in zip(texts, labels):
    label2data[l.strip()].append(t.strip())
for l in label2data:
    label2data[l] = random.sample(label2data[l], 10)

new_texts, new_labels = [], []
for l in label2data:
    new_labels.extend([l] * len(label2data[l]))
    new_texts.extend(label2data[l])

output_dir = "train_10"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, "seq.in"), 'w', encoding="utf-8") as f_text, \
    open(os.path.join(output_dir, "label"), 'w', encoding="utf-8") as f_label:
    f_text.writelines([t.lower() + "\n" for t in new_texts])
    f_label.writelines([l + "\n" for l in new_labels])

# ===== train_5 =====

label2data = defaultdict(list)
for t, l in zip(texts, labels):
    label2data[l.strip()].append(t.strip())
for l in label2data:
    label2data[l] = random.sample(label2data[l], 5)

new_texts, new_labels = [], []
for l in label2data:
    new_labels.extend([l] * len(label2data[l]))
    new_texts.extend(label2data[l])

output_dir = "train_5"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, "seq.in"), 'w', encoding="utf-8") as f_text, \
    open(os.path.join(output_dir, "label"), 'w', encoding="utf-8") as f_label:
    f_text.writelines([t.lower() + "\n" for t in new_texts])
    f_label.writelines([l + "\n" for l in new_labels])