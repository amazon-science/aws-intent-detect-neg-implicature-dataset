"""
The program will use dependency parser to parse user intent from
extracted goals.
Keypoints:
* It is not easy to extract the verb phrase. Sometimes it can be something like "be reminded to" or "listen to".
* The program might not find the object. We will need another prompt to summarize object.
"""
import spacy, re
from tqdm import tqdm
import os, json, argparse
nlp = spacy.load("en_core_web_sm")

def find_verb_phrase(parse):
    pattern = r"<auxpass><ROOT><aux>|<auxpass><ROOT><prep>|<auxpass><ROOT>|<ROOT><prep>|<ROOT>"
    dep_str = ""
    for token in parse:
        dep_str += "<" + token.dep_ + ">"
    match = re.compile(pattern).search(dep_str)
    start, end = match.start(), match.end()
    num_before_match = dep_str[:start].count('<')
    num_in_match = dep_str[start:end].count('<')
    phrase = [t.text for t in parse[num_before_match:num_before_match+num_in_match]]
    root = [t.text for t in parse[num_before_match:num_before_match+num_in_match] if t.dep_ == "ROOT"]
    return ' '.join(phrase), root[0]


def main(args):
    # data path
    assert args.data_path.endswith(".json")
    pred_path = args.data_path.replace(".json", "_parsed.json")
    print(pred_path)
    if os.path.exists(pred_path) and not args.overwrite:
        # I like to save result data into another file
        # so in the end, we have to open both of them
        # and see which predicted one hasn't been parsed yet.
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        for d, pd in zip(data, pred_data):
            if 'action' in pd:
                d['action'] = pd['action']
            if 'object' in pd:
                d['object'] = pd['object']
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    
    # test run, comment it when you are ready
    # import random
    # random.seed(3)
    # data = random.sample(data[:10000], 200)

    for datum in tqdm(data, total=len(data)):
        # if 'action' in datum, then it is already predicted
        # if 'prediction' not in datum, then it is not finished yet
        if 'action' in datum or 'prediction' not in datum:
            continue

        # add this 'I ' here for better parsing
        # parsing is really fast, there is no need to save intermediate results
        doc = nlp('I ' + datum['prediction'])
        datum['action'], root = find_verb_phrase(doc)
        for token in doc[1:]:
            if token.pos_ == "NOUN" and token.dep_ in ["dobj"]:
                if token.head.text == root:
                    datum['object'] = token.text
        # if 'object' not in datum:
        #     for token in doc[1:]:
        #         if token.pos_ == "NOUN" and token.dep_ in ["pobj"]:
        #             if token.head.head.text == root:
        #                 datum['object'] = token.text
    
    with open(pred_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--data_path", type=str, required=True)
    # other
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)