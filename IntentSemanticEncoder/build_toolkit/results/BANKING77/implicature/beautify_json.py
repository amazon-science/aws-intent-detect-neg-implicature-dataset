import json

with open("scenarios_gpt-3.5-turbo-0613_brain_storm_v1.json", 'r') as f:
    data = json.load(f)

with open("scenarios_gpt-3.5-turbo-0613_brain_storm_v1.json", 'w') as f:
    json.dump(data, f, indent=4)