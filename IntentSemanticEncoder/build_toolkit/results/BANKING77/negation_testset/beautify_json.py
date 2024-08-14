import json

with open("modified_utterance_gpt-3.5-turbo-0613_negate_intent_BANKING77_v4_2.json", 'r') as f:
    data = json.load(f)

with open("modified_utterance_gpt-3.5-turbo-0613_negate_intent_BANKING77_v4_2.json", 'w') as f:
    json.dump(data, f, indent=4)