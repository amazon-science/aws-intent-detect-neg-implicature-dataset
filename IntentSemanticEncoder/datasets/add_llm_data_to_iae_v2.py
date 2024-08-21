"""
This version is slightly different from the last one
because we also include the negative data. Furthermore,
the output will be saved in a json file so that the data
can be used in a triplet objective.

This purpose of this code is to construct triplet data using IAE data to validate that adding negative data into it will not affect the performance.
"""
import os, json, random

random.seed(0)

with open("../build_toolkit_with_endpoints/results/pretrain/pretrain_data_v7_paraphrase_252744_proc.json", 'r') as f:
    our_data = json.load(f)

# convert our data to a dict
our_data_dict = {d['utt'].lower(): d for d in our_data}

# load the augmented data
iae_data = []
with open("iae_pretrain_data_llmaug/train.txt", 'r') as f:
    iae_data  += [l.strip().lower().split("||") for l in f.readlines()]
# some of the utterances might not exist in our data
# we have to sample one with different labels
unique_labels = list(set([d[1] for d in iae_data]))
difflab2dict = {}
for ul in unique_labels:
    difflab2dict[ul] = [d[0] for d in iae_data if d[1] != ul]

final_data = []
llm_added_count_train = 0
for datum in iae_data:
    # at least half of these utterances should appear in our list
    # we are not using all the utterances, so there might be some discrepancies
    if datum[0] in our_data_dict:
        hard_negative = our_data_dict[datum[0]]['hard_negative']['results']
        retrieved_negative = our_data_dict[datum[0]]['retrieved_negative']
        if hard_negative and (our_data_dict[datum[0]]['hard_negative']['prompt'] == 'hard_negative_3.txt'):
            llm_added_count_train += 1
            final_datum = {
                "utt": datum[0],
                "pos": [datum[1], datum[2]],
                "neg": hard_negative
            }
            final_data.append(final_datum)
        elif retrieved_negative:
            final_datum = {
                "utt": datum[0],
                "pos": [datum[1], datum[2]],
                "neg": retrieved_negative
            }
            final_data.append(final_datum)
        else:
            final_datum = {
                "utt": datum[0],
                "pos": [datum[1], datum[2]],
                "neg": random.choice(difflab2dict[datum[1]])
            }
            final_data.append(final_datum)
    else:
        final_datum = {
            "utt": datum[0],
            "pos": [datum[1], datum[2]],
            "neg": random.choice(difflab2dict[datum[1]])
        }
        final_data.append(final_datum)

print(llm_added_count_train)
print(len(final_data))
os.makedirs("iae_pretrain_data_llmaug_v2", exist_ok=True)
with open(f"iae_pretrain_data_llmaug_v2/train.json", 'w') as f:
    json.dump(final_data, f, indent=4)

with open("iae_pretrain_data_llmaug_v2/counts.txt", 'w') as f:
    f.write(f"train: {llm_added_count_train}\n")
    f.write(f"total: {len(final_data)}")
