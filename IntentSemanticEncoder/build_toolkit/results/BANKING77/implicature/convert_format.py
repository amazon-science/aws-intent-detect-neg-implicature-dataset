import json

with open("utterances_gpt-3.5-turbo-0613_generate_implicature_v1.json", 'r') as f:
    data = json.load(f)

utterances = []
intents = []
for datum in data:
    for utt in datum['prediction']:
        utterances.append(datum['prediction'][utt]+"\n")
        intents.append(datum['label']+"\n")

with open("utterances_gpt-3.5-turbo-0613_generate_implicature_v1.in", 'w') as f:
    f.writelines(utterances)
with open("utterances_gpt-3.5-turbo-0613_generate_implicature_v1_label", 'w') as f:
    f.writelines(intents)