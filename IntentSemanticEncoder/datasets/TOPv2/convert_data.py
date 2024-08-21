import os, csv

data_folder = "TOPv2"
texts = []
for file_name in os.listdir(data_folder):
    if file_name.endswith(".tsv"):
        with open(os.path.join(data_folder, file_name), 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            texts.extend([l[1] for l in csv_reader if l[0] != "domain"])


with open("train/seq.in", 'w', encoding='utf-8') as f:
    f.writelines([t.strip().lower() + '\n' for t in texts])