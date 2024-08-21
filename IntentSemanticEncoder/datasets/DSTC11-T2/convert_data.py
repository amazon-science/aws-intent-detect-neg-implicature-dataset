import os, json

texts = []
data_folder = "dstc11-track2-intent-induction/dstc11"

for split in ['development', 'test-banking', 'test-finance']:
    with open(os.path.join(data_folder, f"{split}/dialogues.jsonl"), 'r') as f:
        data = [json.loads(l) for l in f]
    for diag in data:
        for turn in diag['turns']:
            if turn['speaker_role'] == 'Customer':
                texts.append(turn['utterance'])
    with open(os.path.join(data_folder, f"{split}/test-utterances.jsonl"), 'r') as f:
        data = [json.loads(l) for l in f]
    for datum in data:
        texts.append(datum['utterance'])

with open("train/seq.in", 'w', encoding='utf-8') as f:
    f.writelines([t.strip().lower() + "\n" for t in texts])
