import os, csv

data_dir = "task-specific-datasets/banking_data"

assert os.path.exists(data_dir)

# ===== test =====

with open(os.path.join(data_dir, "test.csv"), 'r') as f:
    csv_reader = csv.reader(f)
    data = [l for l in csv_reader if not (l[0] == "text" and l[1] == "category")]

output_dir = "test"

assert os.path.exists(output_dir)

with open(os.path.join(output_dir, "seq.in"), 'w', encoding="utf-8") as f_text, \
    open(os.path.join(output_dir, "label"), 'w', encoding="utf-8") as f_label:
    f_text.writelines([d[0].strip().lower() + "\n" for d in data])
    f_label.writelines([d[1].strip() + "\n" for d in data])

# ===== train =====

with open(os.path.join(data_dir, "train.csv"), 'r') as f:
    csv_reader = csv.reader(f)
    data = [l for l in csv_reader if not (l[0] == "text" and l[1] == "category")]

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