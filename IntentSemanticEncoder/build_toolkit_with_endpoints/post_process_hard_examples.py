"""
This program is to
* clean up repetitive utterances
* diversify hard examples by regex
* filter those that has 'ai language model' (these are definitely not utterances)
"""
import json, os, argparse, random, re
from tqdm import tqdm
import spacy, pyinflect
random.seed(0)

nlp = spacy.load('en_core_web_sm')

def post_process_hard_positive_4(utt: str, original_utt: str):
    """
    ...
    Imagine you are the customer. Tell me 3 other kinds of [OBJECT] that you want to [ACTION]. And use different language styles to express you needs.
    ...
    """
    if utt == original_utt:
        return None
    if 'ai language model' in utt:
        return None
    want_to_alternatives = ("i want to ", "i would like to ", "could you help me ", "i need to ", \
                            "want to ", "would like to ", "need to ", "could you please help me ")
    if utt.startswith(want_to_alternatives):
        for wta in want_to_alternatives:
            if utt.startswith(wta):
                # for these, you do not want to remove if completely
                if not utt.startswith((wta+"know", wta+"be")):
                    new_start = random.choice(want_to_alternatives)
                else:
                    if random.random() > 0.2:
                        new_start = random.choice(want_to_alternatives)
                    else:
                        new_start = ""
                break
        utt = utt.replace(wta, new_start)
    if utt.startswith("i am "):
        if random.random() > 0.5:
            utt = utt.replace("i am ", "")
    return utt

def post_process_hard_positive_5(utt: str, original_utt: str):
    """
    ...
    Imagine you are the customer. Give me 1 other ways to express what you want.
    ...
    """
    if utt == original_utt:
        return None
    if 'ai language model' in utt:
        return None
    return utt

def post_process_hard_positive_6(utt: str, original_utt: str):
    """
    ...
    Imagine you are the customer. Give me 1 reasons that you want to do this.
    ...
    """
    if utt == original_utt:
        return None
    if 'ai language model' in utt:
        return None
    want_to_alternatives = ("i want to ", "i would like to ", "could you help me ", "i need to ", \
                            "want to ", "would like to ", "need to ", "could you please help me ")
    if utt.startswith(want_to_alternatives):
        for wta in want_to_alternatives:
            if utt.startswith(wta):
                # for these, you do not want to remove if completely
                if not utt.startswith((wta+"know", wta+"be")):
                    new_start = random.choice(want_to_alternatives)
                else:
                    if random.random() > 0.2:
                        new_start = random.choice(want_to_alternatives)
                    else:
                        new_start = ""
                break
        utt = utt.replace(wta, new_start)
    if utt.startswith("i am "):
        if random.random() > 0.5:
            utt = utt.replace("i am ", "")
    return utt

def post_process_hard_positive_7(utt: str, original_utt: str):
    """
    ...
    Imagine you are the customer. Give me 2 things that you do not want to do in this scenario.
    ...
    """
    if utt == original_utt:
        return None
    if 'ai language model' in utt:
        return None
    if utt.startswith("i do not want to "):
        rnd = random.random()
        if rnd > 0.5:
            alternatives = ("i try not to ", "i'd rather not to ", "i prefer not to ", "could you not let me ", "i would rather not to ", "do not let me ", "never let me ")
            utt = utt.replace("i do not want to ", random.choice(alternatives))
        else:
            alternatives = ("i try to avoid ", "i'm not interested in ", "i'm not into ", "i'd rather avoid ", "i am not interested in ", "i am not looking for ", \
                            "i am not into ", "i would rather avoid ", "i'm not looking for ", "no ", "is there a way to avoid ")
            utt = utt.replace("i do not want to ", "")
            doc = nlp(utt)
            for idx in range(len(doc)):
                if doc[idx].pos_ == "VERB":
                    vbg = doc[idx]._.inflect('VBG', inflect_oov=True)
                    if vbg is not None:
                        utt = utt.replace(doc[idx].text, vbg, 1)
                    break
            utt = random.choice(alternatives) + utt
    return utt

def post_process_hard_negative_2(utt: str, original_utt: str):
    """
    ...
    Imagine you are a customer. Give me 2 other things you want to [ACTION] rather than [OBJECT]. Respond with sentences in a similar style as above.
    ...
    """
    if utt == original_utt:
        return None
    if 'ai language model' in utt:
        return None
    return utt

def post_process_hard_negative_3(utt: str, original_utt: str):
    """
    ...
    Imagine you are the customer. Now you no longer need to [ACTION] [OBJECT]. Give me 2 reasons for that.
    ...
    """
    if utt == original_utt:
        return None
    if 'ai language model' in utt:
        return None
    # find those that do not have "not"
    pattern = r"the customer may (?!not)"
    match = re.search(pattern, utt)
    if match is not None:
        alternatives = ("i ", "", "i may ")
        utt = re.sub(pattern, random.choice(alternatives), utt)
    elif "the customer may not " in utt:
        alternatives = ("i do not ", "i may not ")
        utt = utt.replace("the customer may not ", random.choice(alternatives))
    else:
        utt = utt.replace("the customer ", "i ")
    return utt

def post_process_hard_negative_4(utt: str, original_utt: str):
    """
    ...
    Imagine you are a customer. Now you do not want to [ACTION] [OBJECT], give me 2 other things you want to do.
    ...
    """
    if utt == original_utt:
        return None
    if 'ai language model' in utt:
        return None
    want_to_alternatives = ("i want to ", "i would like to ", "could you help me ", "i need to ", \
                            "want to ", "would like to ", "need to ", "could you please help me ")
    if utt.startswith(want_to_alternatives):
        for wta in want_to_alternatives:
            if utt.startswith(wta):
                # for these, you do not want to remove if completely
                if not utt.startswith((wta+"know", wta+"be")):
                    new_start = random.choice(want_to_alternatives)
                else:
                    if random.random() > 0.2:
                        new_start = random.choice(want_to_alternatives)
                    else:
                        new_start = ""
                break
        utt = utt.replace(wta, new_start)
    return utt

PROMPT2FUNC = {
    "hard_positive_4.txt": post_process_hard_positive_4,
    "hard_positive_5.txt": post_process_hard_positive_5,
    "hard_positive_6.txt": post_process_hard_positive_6,
    "hard_positive_7.txt": post_process_hard_positive_7,
    "hard_negative_2.txt": post_process_hard_negative_2,
    "hard_negative_3.txt": post_process_hard_negative_3,
    "hard_negative_4.txt": post_process_hard_negative_4
}

def main(args):
    assert args.data_path.endswith(".json")
    pred_path = args.data_path.replace(".json", f"_proc.json")
    print(pred_path)
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    
    # data = data[:5000]

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if 'hard_positive' in datum:
            prompt_name = datum['hard_positive']['prompt']
            post_process_func = PROMPT2FUNC[prompt_name]
            post_processed_list = []
            for utt in datum['hard_positive']['results']:
                pp_utt = post_process_func(utt, datum['utt'])
                if pp_utt is not None:
                    post_processed_list.append(pp_utt)
            datum['hard_positive']['results'] = post_processed_list
        if 'hard_negative' in datum:
            prompt_name = datum['hard_negative']['prompt']
            post_process_func = PROMPT2FUNC[prompt_name]
            post_processed_list = []
            for utt in datum['hard_negative']['results']:
                pp_utt = post_process_func(utt, datum['utt'])
                if pp_utt is not None:
                    post_processed_list.append(pp_utt)
            datum['hard_negative']['results'] = post_processed_list

    with open(pred_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--data_path", type=str, required=True)
    # other
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)