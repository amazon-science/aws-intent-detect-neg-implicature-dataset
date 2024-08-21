# merge all pre-training datasets and check for overlaps

all_texts = set()
for dataset in ["MultiWOZ", "SGD", "TOP", "TOPv2", "DSTC11-T2"]:
    with open(f"../{dataset}/train/seq.in", 'r') as f:
        for l in f.readlines():
            all_texts.add(l)

with open("seq.in", 'w', encoding='utf-8') as f:
    f.writelines(list(all_texts))