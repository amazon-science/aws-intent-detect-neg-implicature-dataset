# We only need the user utterances
import os, json


texts = []
for split in ['train', 'dev', 'test']:
    data_folder = f"multiwoz/data/MultiWOZ_2.2/{split}"
    file_paths = [os.path.join(data_folder, file_name) for file_name in os.listdir(data_folder)]
    for fp in file_paths:
        with open(fp, 'r') as f:
            data = json.load(f)
        
        for diag in data:
            for turn in diag['turns']:
                if turn['speaker'] == 'USER':
                    texts.append(turn['utterance'])

output_dir = "train"
with open(os.path.join(output_dir, 'seq.in'), 'w', encoding='utf-8') as f_text:
    f_text.writelines([t.strip().lower() + '\n' for t in texts])
