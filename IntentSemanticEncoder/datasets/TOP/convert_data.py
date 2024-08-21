import os, csv


texts = []
for split in ['train', 'eval', 'test']:
    file_path = os.path.join("top-dataset-semantic-parsing", split+'.tsv')
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter="\t")
        texts.extend([l[0].strip().lower() + "\n" for l in csv_reader])


with open("train/seq.in", 'w', encoding='utf-8') as f:
    f.writelines(texts)