"""
Before generating any implicatures, generate the scenarios for each intent.
"""
import openai
import os, argparse, json
from tqdm import tqdm
from openai_tools import delayed_completion

def load_intent_examples(original_label_path, domain_path):

    # label files are organized so that each line is a single label.
    # each line has two columns separated by coma, where former is the intent index
    # and the latter is the intent name or corresponding domain
    with open(original_label_path, 'r', encoding="utf-8") as f_exp:
        exps = [l.strip().split(",") for l in f_exp.readlines()]
        lab2exp = {l[0]:l[1] for l in exps}
    
    with open(domain_path, 'r', encoding="utf-8") as f_dom:
        lines = [l.strip().split(",") for l in f_dom.readlines()]
        lab2dom = {l[0]:l[1].replace("_", " ") for l in lines}

    return lab2exp, lab2dom

def prepare_input(prompt: str, intent: str, domain: str, num: int):
    """
    domain is usually services like 'banking', 'automobile', 'dining'
    num is how many scenarios you want to generate
    """
    num = str(num)
    prepared = prompt.replace("[INTENT]", intent)
    prepared = prepared.replace("[DOMAIN]", domain)
    prepared = prepared.replace("[NUM]", num)
    return prepared

def parse_completion(completion, num: int):
    """
    All prompts for modification should instruct the LLM to wrap output
    within JSON object. Find the content through output key.
    """
    content = completion['choices'][0]['message']['content'].strip()
    assert isinstance(content, str)
    try:
        results = json.loads(content)
        assert isinstance(results, dict)
        for i in range(num):
            assert f"scenario_{i+1}" in results
    except Exception as e:
        # you will have to parse it manually here
        print(e)
        # print(completion)
        print(content)
        results = {}
    return content, results

def main(args):
    openai.organization = os.getenv("OPENAI_ORG_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(args.prompt_path, 'r') as f:
        prompt = ''.join(f.readlines())
    
    # get file name of prompt
    prompt_name = os.path.splitext(args.prompt_path)[0].split("/")[-1]

    os.makedirs(f"build_toolkit/results/{args.dataset}/implicature", exist_ok=True)
    pred_path = os.path.join(f"build_toolkit/results/{args.dataset}/implicature/scenarios_{args.model_name}_{prompt_name}.json")
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        lab2exp, lab2dom = load_intent_examples(args.original_label_path, args.domain_path)
        prepared = \
            [prepare_input(prompt, lab2exp[lab], lab2dom[lab], args.num) for lab in lab2exp.keys()]
        data = \
            [{"label": lab, "intent": lab2exp[lab], "domain": lab2dom[lab], "prepared": prep} \
             for lab, prep in zip(lab2exp.keys(), prepared)]
    
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print(datum['prepared'])
            # breakpoint()
        if 'prediction' in datum or not datum['intent']:
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
            content, results = parse_completion(completion, args.num)
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
    parser.add_argument("--original_label_path", type=str, required=True)
    parser.add_argument("--domain_path", type=str, required=True)
    # llm
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.,
                        help="https://platform.openai.com/docs/guides/gpt/why-are-model-outputs-inconsistent")
    parser.add_argument("--max_token", type=int, default=768)
    # other
    parser.add_argument("--dataset", type=str, default="BANKING77")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)