"""
In order to show the potential of llm data. We add llm generated positive data into iae models.

However, since iae does not utilize negative data in their objective. There is no way to add negative data into it.
That being said, we can still hope to observe an improvement beyond triplet tasks.

Several factors to be considered:
* use the utterance as an index and search for the llm augmentation in our file.
* keep the original format so that there is nothing to change in the code.
"""
import os, json, random

random.seed(0)

with open("../build_toolkit_with_endpoints/results/pretrain/pretrain_data_v7_paraphrase_252744_proc.json", 'r') as f:
    our_data = json.load(f)

# convert our data to a dict
our_data_dict = {d['utt'].lower(): d for d in our_data}

iae_data = []
with open("iae_pretrain_data/train.txt", 'r') as f:
    iae_data  += [l.strip().lower().split("||") for l in f.readlines()]

final_data = []
llm_added_count_train = 0
for datum in iae_data:
    # at least half of these utterances should appear in our list
    # we are not using all the utterances, so there might be some discrepancies
    if datum[0] in our_data_dict:
        hard_positive = our_data_dict[datum[0]]['hard_positive']['results']
        if hard_positive and (our_data_dict[datum[0]]['hard_positive']['prompt'] != 'hard_positive_7.txt'):
            llm_added_count_train += 1
            # replace the last one so that we do not have to change the code
            datum[-1] = random.choice(hard_positive)

print(llm_added_count_train)
os.makedirs("iae_pretrain_data_llmaug", exist_ok=True)
with open(f"iae_pretrain_data_llmaug/train.txt", 'w') as f:
    iae_data = ['||'.join(datum)+'\n' for datum in iae_data]
    f.writelines(iae_data)

# ----- val -----
iae_data = []
with open("iae_pretrain_data/val.txt", 'r') as f:
    iae_data  += [l.strip().lower().split("||") for l in f.readlines()]

final_data = []
llm_added_count_val = 0
for datum in iae_data:
    # at least half of these utterances should appear in our list
    # we are not using all the utterances, so there might be some discrepancies
    if datum[0] in our_data_dict:
        hard_positive = our_data_dict[datum[0]]['hard_positive']['results']
        if hard_positive and (our_data_dict[datum[0]]['hard_positive']['prompt'] != 'hard_positive_7.txt'):
            llm_added_count_val += 1
            # replace the last one so that we do not have to change the code
            datum[-1] = random.choice(hard_positive)

print(llm_added_count_val)
os.makedirs("iae_pretrain_data_llmaug", exist_ok=True)
with open(f"iae_pretrain_data_llmaug/val.txt", 'w') as f:
    iae_data = ['||'.join(datum)+'\n' for datum in iae_data]
    f.writelines(iae_data)

with open("iae_pretrain_data_llmaug/counts.txt", 'w') as f:
    f.write(f"train: {llm_added_count_train}\n")
    f.write(f"val: {llm_added_count_val}")