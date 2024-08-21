# This code is for splitting data into a separate dev set so that we can develop the model on that set
import os, random
from collections import defaultdict

random.seed(0)

dataset = "BANKING77"

with open(f"{dataset}/train/seq.in", 'r') as f_text, open(f"{dataset}/train/label", 'r') as f_label:
    texts = [t for t in f_text.readlines()]
    labels = [l for l in f_label.readlines()]

with open(f"{dataset}/train_10/seq.in", 'r') as f_text, open(f"{dataset}/train_10/label", 'r') as f_label:
    texts_10 = [t for t in f_text.readlines()]
    labels_10 = [l for l in f_label.readlines()]

# excluding the 10-shot ones because they are used for acquiring prototypes
rest_texts = []
rest_labels = []
label2data = defaultdict(list)
for t, l in zip(texts, labels):
    if t not in texts_10:
        label2data[l].append(t)

upper = 30
for l in label2data:
    if len(label2data[l]) > upper:
        label2data[l] = random.sample(label2data[l], upper)

for l in label2data:
    rest_texts += label2data[l]
    rest_labels += [l] * len(label2data[l])

os.makedirs(f"{dataset}/dev", exist_ok=True)

with open(f"{dataset}/dev/seq.in", 'w') as f:
    f.writelines(rest_texts)
with open(f"{dataset}/dev/label", 'w') as f:
    f.writelines(rest_labels)

print("num data: ", len(rest_labels))
print("num labels: ", len(set(rest_labels)))