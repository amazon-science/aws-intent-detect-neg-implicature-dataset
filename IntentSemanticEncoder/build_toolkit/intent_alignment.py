"""
Implicature utterances are multi-label.
This file tests chatgpt's ability on implicature set for multi-class classification. So that we know the upper bound.
Take the top-5 classes + 1 ground truth from 10-shot classification, and then find the most probable choice by prompting. Check how often the llm will select ground truth from the six candidates.
"""
import openai
import os, argparse, json, random
from tqdm import tqdm
from openai_tools import delayed_completion

random.seed(0)

def prepare_input(prompt: str, intents: str, utterance: str):
    prepared = prompt.replace("[UTTERANCE]", utterance)
    intents_str = ','.join(intents)
    prepared = prepared.replace("[INTENT]", intents_str)
    return prepared

def parse_completion(completion):
    """
    All prompts for modification should instruct the LLM to wrap output
    within JSON object. Find the content through output key.
    """
    content = completion['choices'][0]['message']['content'].strip()
    assert isinstance(content, str)
    try:
        results = json.loads(content)
        assert isinstance(results, dict)
        results = results['intent']
    except Exception as e:
        # you will have to parse it manually here
        print(e)
        # print(completion)
        print(content)
        results = {}
    return content, results

def load_intent_examples(file_path=None, seq_file_path=None, label_file_path=None):
    texts, labels = [], []
    # text files are organized so that texts and labels are saved
    # in separate files. Use the line index to correspond two.
    # it seems that some encoders only accept lower case, here we
    # only use lower case for all encoders.
    if file_path is not None:
        with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
            for text, label in zip(f_text, f_label):
                texts.append(text.strip().lower())
                labels.append(label.strip())
    elif seq_file_path is not None and label_file_path is not None:
        with open(seq_file_path, 'r', encoding="utf-8") as f_text, open(label_file_path, 'r', encoding="utf-8") as f_label:
            for text, label in zip(f_text, f_label):
                texts.append(text.strip().lower())
                labels.append(label.strip())
    else:
        raise ValueError("You need to provide one of the paths!")
    return texts, labels

def load_label_file(file_path):
    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip().split(",") for l in lines]
    labels = {l[0]: l[1].replace("_", " ").lower() if len(l) == 2 else l[0].replace("_", " ").lower() for l in lines}
    return labels

def main(args):
    openai.organization = os.getenv("OPENAI_ORG_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(args.prompt_path, 'r') as f:
        prompt = ''.join(f.readlines())
    
    # get file name of prompt
    prompt_name = os.path.splitext(args.prompt_path)[0].split("/")[-1]

    os.makedirs(f"build_toolkit/results/{args.dataset}/intent_align", exist_ok=True)
    pred_path = os.path.join(f"build_toolkit/results/{args.dataset}/intent_align/{args.model_name}_{prompt_name}.json")
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        # this will be a dict that use gt as keys and intent explanations as values
        candidates = load_label_file(args.candidate_path)
        lab_txt = [candidates[k] for k in sorted(list(candidates.keys()))]
        with open(args.log_path, 'r') as f:
            ranks = json.load(f)['rank']
        utt, lab = load_intent_examples(file_path=args.test_data_path, seq_file_path=args.test_data_path_seqin, label_file_path=args.test_data_path_label)

        # get candidate set for each utterance
        data = []
        for u, l, rs in zip(utt, lab, ranks):
            cands = [lab_txt[r] for r in rs[:args.topk]]
            cands.append(candidates[l])
            cands = list(set(cands))
            random.shuffle(cands)
            prepared = prepare_input(prompt, cands, u)
            data.append({
                "utt": u,
                "lab": candidates[l],
                "class_pred": lab_txt[rs[0]],
                "candidates": cands,
                "prepared": prepared
            })
    
    # data = random.sample(data, 20)
    
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print(datum['prepared'])
            # breakpoint()
        if 'prediction' in datum:
            continue
        messages = [
            {"role": "user", "content": datum['prepared']}
        ]
        completion, error = \
            delayed_completion(
                delay_in_seconds=args.delay,
                max_trials=args.max_trials,
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_token,
                temperature=args.temperature
        )
        if completion is None:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(error)
            breakpoint() # for debugging
        else:
            content, results = parse_completion(completion)
            data[idx]['content'] = content # for debugging the parsing
            data[idx]['prediction'] = results
            # breakpoint()
        
        # save intermediate results to avoid any termination of program
        if idx % args.save_every == 0 and idx > 0:
            print(f"Save data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                json.dump(data, f, indent=4)
    
    with open(pred_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--candidate_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_data_path_seqin", type=str, default=None)
    parser.add_argument("--test_data_path_label", type=str, default=None)
    parser.add_argument("--log_path", type=str, required=True)
    # llm
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.,
                        help="https://platform.openai.com/docs/guides/gpt/why-are-model-outputs-inconsistent")
    parser.add_argument("--max_token", type=int, default=512)
    # other
    parser.add_argument("--dataset", type=str, default="BANKING77")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)