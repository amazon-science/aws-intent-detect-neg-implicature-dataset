import os, json

data_path = "oos-eval/data/data_full.json"

assert os.path.exists(data_path)

with open(data_path, 'r') as f:
    data_full = json.load(f)

assert len(data_full) == 6

# ===== test =====

assert 'test' in data_full

data = data_full['test']

output_dir = "test"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, "seq.in"), 'w', encoding="utf-8") as f_text, \
    open(os.path.join(output_dir, "label"), 'w', encoding="utf-8") as f_label:
    f_text.writelines([d[0].strip().lower() + "\n" for d in data])
    f_label.writelines([d[1].strip() + "\n" for d in data])

# ===== train =====

assert "train" in data_full

data = data_full['train']

output_dir = "train"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, "seq.in"), 'w', encoding="utf-8") as f_text, \
    open(os.path.join(output_dir, "label"), 'w', encoding="utf-8") as f_label:
    f_text.writelines([d[0].strip().lower() + "\n" for d in data])
    f_label.writelines([d[1].strip() + "\n" for d in data])

# ===== train_10 =====

import random
from collections import defaultdict
random.seed(0)

label2data = defaultdict(list)
for d in data:
    label2data[d[1].strip()].append(d[0].strip())
for l in label2data:
    label2data[l] = random.sample(label2data[l], 10)

texts, labels = [], []
for l in label2data:
    labels.extend([l] * len(label2data[l]))
    texts.extend(label2data[l])

output_dir = "train_10"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, "seq.in"), 'w', encoding="utf-8") as f_text, \
    open(os.path.join(output_dir, "label"), 'w', encoding="utf-8") as f_label:
    f_text.writelines([t.lower() + "\n" for t in texts])
    f_label.writelines([l + "\n" for l in labels])

# ===== train_5 =====

label2data = defaultdict(list)
for d in data:
    label2data[d[1].strip()].append(d[0].strip())
for l in label2data:
    label2data[l] = random.sample(label2data[l], 5)

texts, labels = [], []
for l in label2data:
    labels.extend([l] * len(label2data[l]))
    texts.extend(label2data[l])

output_dir = "train_5"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, "seq.in"), 'w', encoding="utf-8") as f_text, \
    open(os.path.join(output_dir, "label"), 'w', encoding="utf-8") as f_label:
    f_text.writelines([t.lower() + "\n" for t in texts])
    f_label.writelines([l + "\n" for l in labels])