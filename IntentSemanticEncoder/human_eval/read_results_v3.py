"""
* calculate inter-annotator agreement
* calculate human-machine agreement
* accomodate splits
* accomodate different kinds of answers
"""
import os, argparse, csv, json
from collections import defaultdict, Counter

def main(args):
    data_path = f"human_eval/results/{args.dataset}"
    assert os.path.exists(data_path)
    anno_path = [os.path.join(data_path, f"annotation_{i+1}") for i in range(3)]

    all_annos = defaultdict(list)
    for ap in anno_path:
        assert os.path.exists(ap), f"{ap} not exists"

        subdirs = os.listdir(ap)
        for sd in subdirs:
            sdp = os.path.join(ap, sd)
            if os.path.isdir(sdp):

                csv_file = [filename for filename in os.listdir(sdp) if filename.endswith(".csv")]
                assert len(csv_file) == 1
                csv_file = csv_file[0]

                with open(os.path.join(sdp, csv_file), 'r') as f:
                    csv_reader = csv.reader(f)
                    annos = [l for l in csv_reader][1:]
                with open(os.path.join(sdp, "types.txt"), 'r') as f:
                    types = [l.strip().split(",")[-1] for l in f.readlines()]
                
                for a, t in zip(annos, types):
                    if a[-2] in ['T', 'F', 'TRUE', 'FALSE', 'Y', 'N']:
                        correct_intent = a[-2] in ['T', 'Y', 'TRUE']
                    else:
                        correct_intent = None
                    if correct_intent == False:
                        implicature = None
                    elif correct_intent == True and a[-1] in ['T', 'F', 'TRUE', 'FALSE', 'Y', 'N']:
                        implicature = a[-1] in ['F', 'N', 'FALSE']
                    else:
                        implicature = None
                    index = a[0].split('_')[0]

                    # collect via index
                    all_annos[index].append((correct_intent, implicature, t))
    
    inter_annotator_correct_intent = []
    inter_annotator_implicature = []
    hm_agree_correct_intent = defaultdict(list)
    hm_agree_implicature = defaultdict(list)
    for index in all_annos:
        correct_intent_list = [tp[0] for tp in all_annos[index] if tp[0] is not None]
        if len(correct_intent_list) == 3:
            inter_annotator_correct_intent.append(len(set(correct_intent_list)) == 1)
            # if inter_annotator_correct_intent[-1] == False:
            #     print(index)
            # human choice is the majority vote
            human_choice = Counter(correct_intent_list).most_common(1)[0][0]
            # machine choice is whether this utterance belong to 'negation'
            # breakpoint()
            hm_agree_correct_intent['total'].append(human_choice == (all_annos[index][2][2] != 'negation'))
            hm_agree_correct_intent[all_annos[index][2][2]].append(human_choice == (all_annos[index][2][2] != 'negation'))
            # if hm_agree_correct_intent[-1] == False:
            #     print(index)
        implicature_list = [tp[1] for tp in all_annos[index] if tp[1] is not None]
        if len(implicature_list) == 3:
            inter_annotator_implicature.append(len(set(implicature_list)) == 1)
            # if inter_annotator_implicature[-1] == False:
            #     print(index)
            # human choice is the majority vote
            human_choice = Counter(implicature_list).most_common(1)[0][0]
            # machine choice is whether this utterance belong to 'implicature'
            hm_agree_implicature['total'].append(human_choice == (all_annos[index][2][2] == 'implicature'))
            hm_agree_implicature[all_annos[index][2][2]].append(human_choice == (all_annos[index][2][2] == 'implicature'))
            # if hm_agree_implicature[-1] == False:
            #     print(index)
        
    logs = {
        "inter_annotator_agreement_correct_intent": (len(inter_annotator_correct_intent), sum(inter_annotator_correct_intent)),
        "inter_annotator_agreement_implicature": (len(inter_annotator_implicature), sum(inter_annotator_implicature)),
    }
    for k in hm_agree_correct_intent:
        logs[f'human_machine_agreement_correct_intent_{k}'] = (len(hm_agree_correct_intent[k]), sum(hm_agree_correct_intent[k]))
    for k in hm_agree_implicature:
        logs[f'human_machine_agreement_implicature_{k}'] = (len(hm_agree_implicature[k]), sum(hm_agree_implicature[k]))

    print(logs)
    save_path = f"human_eval/results/{args.dataset}/logs.json"
    with open(save_path, 'w') as f:
        json.dump(logs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BANKING77")
    args = parser.parse_args()

    main(args)