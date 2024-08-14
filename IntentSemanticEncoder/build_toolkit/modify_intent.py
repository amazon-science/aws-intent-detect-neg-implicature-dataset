"""
Modify intent of an utterance with least modifications
possible. Provide a prompt and target intents.
"""
import openai
import os, argparse, json
from tqdm import tqdm
from openai_tools import delayed_completion

def load_intent_examples(original_label_path, modified_label_path, file_path):
    texts, original_labels, modified_labels = [], [], []

    # label files are organized so that each line is a single label.
    # each line has two columns separated by coma, where former is the intent index
    # and the latter is the original intent name or modified intent name
    with open(original_label_path, 'r', encoding="utf-8") as f_exp:
        exps = [l.strip().split(",") for l in f_exp.readlines()]
        lab2exp = {l[0]:l[1] for l in exps}
    with open(modified_label_path, 'r', encoding="utf-8") as f_mod:
        mods = [l.strip().split(",") for l in f_mod.readlines()]
        lab2mod = {l[0]:l[1] for l in mods}

    # text files are organized so that texts and labels are saved
    # in separate files. Use the line index to correspond two.
    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            texts.append(text.strip().lower())
            lab = label.strip()
            assert lab in lab2exp
            original_labels.append(lab2exp[lab])
            modified_labels.append(lab2mod[lab])

    return texts, original_labels, modified_labels

def prepare_input(prompt: str, utterance: str, original_intent: str, new_intent: str):
    prepared = prompt.replace("[UTTERANCE]", utterance)
    prepared = prepared.replace("[ORIGINAL_INTENT]", original_intent)
    prepared = prepared.replace("[NEW_INTENT]", new_intent)
    return prepared

def parse_completion(completion):
    """
    All prompts for modification should instruct the LLM to wrap output
    within JSON object. Find the content through output key.
    """
    content = completion['choices'][0]['message']['content'].strip()
    assert isinstance(content, str)
    try:
        # start_idx, end_idx = content.index("<"), content.index(">")
        # results = content[start_idx+1:end_idx]
        results = json.loads(content)['output']
        results = results.replace("\n", " ").replace("\t", " ")
    except Exception as e:
        # you will have to parse it manually here
        print(e)
        # print(completion)
        print(content)
        results = ""
    return content, results

def main(args):
    openai.organization = os.getenv("OPENAI_ORG_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(args.prompt_path, 'r') as f:
        prompt = ''.join(f.readlines())
    
    # get file name of prompt
    prompt_name = os.path.splitext(args.prompt_path)[0].split("/")[-1]

    os.makedirs(f"build_toolkit/results/{args.dataset}/{args.challenge}", exist_ok=True)
    pred_path = os.path.join(f"build_toolkit/results/{args.dataset}/{args.challenge}/modified_utterance_{args.model_name}_{prompt_name}.json")
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        texts, original_labels, modified_labels = \
            load_intent_examples(args.original_label_path, args.modified_label_path, args.file_path)
        prepared = \
            [prepare_input(prompt, text, ol, ml) for text, ol, ml in zip(texts, original_labels, modified_labels)]
        data = \
            [{"text": text, "original_label": ol, "modified_label": ml, "prepared": prep} \
             for text, ol, ml, prep in zip(texts, original_labels, modified_labels, prepared)]
    
    # test run, comment it when you are ready
    # import random
    # random.seed(3)
    # data = random.sample(data, 50)

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print(datum['prepared'])
            # breakpoint()
        if not datum['modified_label']:
            data[idx]['content'] = ""
            data[idx]['prediction'] = ""
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
    
    # This is just for easy to use
    with open(pred_path.replace(".json", ".in"), 'w') as f:
        f.writelines([d['prediction'] + "\n" for d in data])
    
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True,
                        help="Save utterances and labels.")
    parser.add_argument("--original_label_path", type=str, required=True)
    parser.add_argument("--modified_label_path", type=str, required=True)
    # llm
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.,
                        help="https://platform.openai.com/docs/guides/gpt/why-are-model-outputs-inconsistent")
    parser.add_argument("--max_token", type=int, default=256)
    # other
    parser.add_argument("--dataset", type=str, default="BANKING77")
    parser.add_argument("--challenge", type=str, default="negation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)