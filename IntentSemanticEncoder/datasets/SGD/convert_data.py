import os, json


texts = []
for split in ['train', 'dev', 'test']:
    data_folder = f"dstc8-schema-guided-dialogue/{split}"
    file_paths = [os.path.join(data_folder, file_name) for file_name in os.listdir(data_folder) if not file_name == "schema.json"]
    for fp in file_paths:
        with open(fp, 'r') as f:
            data = json.load(f)
        
        for diag in data:
            try:
                for turn in diag['turns']:
                    if turn['speaker'] == 'USER':
                        texts.append(turn['utterance'])
            except:
                print(fp)

output_dir = "train"
with open(os.path.join(output_dir, 'seq.in'), 'w', encoding='utf-8') as f_text:
    f_text.writelines([t.strip().lower() + '\n' for t in texts])