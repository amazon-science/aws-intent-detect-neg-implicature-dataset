# a universal way to generate hard examples.
# modified from `modify_intents.py`
import os, argparse, json, string, random, time
from tqdm import tqdm
from typing import List
from endpoints_tools import setup, MODEL2ENDPOINTS

random.seed(0)

def prepare_input(prompts: List[str], previous_prompt: str, previous_pred: str, action: str, object: str, utt: str):
    prompt_file, prompt = random.choice(prompts)
    if action in ['change', 'cancel', 'decline', 'reject', 'refuse', 'negate'] and prompt_file == "hard_negative_3.txt":
        # falcon-40b-instruct is not able to handle double negations
        # e.g. do not need to decline --> need to accept
        # simply sample another prompt in this situation
        # maybe enriched later
        while prompt_file == "hard_negative_3.txt":
            prompt_file, prompt = random.choice(prompts)
    prepared = prompt.replace("[PREVIOUS]", previous_prompt)
    prepared = prepared.replace("[PRED]", previous_pred)
    prepared = prepared.replace("[ACTION]", action)
    prepared = prepared.replace("[OBJECT]", object)
    prepared = prepared.replace("[UTTERANCE]", utt)
    return prompt_file, prepared

def parse_completion(response, model_name: str, prompt: str):
    """
    Unfortunately, the parsing function has to be different for each model.
    """
    output = response['Body'].read().decode('utf-8')
    output = json.loads(output)[0]
    try:
        if model_name in ["mpt-30b-chat", "falcon-40b-instruct"]:
            if model_name == "falcon-40b-instruct":
                output = output['generated_text']
            prompt_len = len(prompt)
            results = output[prompt_len:].strip().lower()
            if '1)' not in prompt:
                results = results.split("\n\n")[0]
                out_list = [results]
            else:
                out_list = []
                after = results
                for ind_str in ['2)', '3)']:
                    # assert ind_str in after, f"{ind_str} not in {after}"
                    if ind_str in after:
                        before, after = after.split(ind_str, 1)
                        out_list.append(before.strip())
                before = after.split("\n")[0]
                out_list.append(before.strip())
                # assert len(out_list) == 3
                assert len(out_list) > 0
    except Exception as e:
        # you will have to parse it manually here
        print(e)
        # print(completion)
        print(output)
        out_list = []
    return output, out_list

def main(args):
    assert args.mode in ['hard_negative', 'hard_positive']

    prompts = []
    for filename in os.listdir("build_toolkit_with_endpoints/prompts"):
        if args.mode in filename:
            with open(os.path.join("build_toolkit_with_endpoints/prompts", filename), 'r') as f:
                prompts.append((filename, ''.join(f.readlines())))

    os.makedirs(f"build_toolkit_with_endpoints/results/pretrain", exist_ok=True)
    assert args.data_path.endswith(".json")
    pred_path = args.data_path.replace(".json", f"_{args.mode}.json")
    print(pred_path)
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    
        # hard_positive is predicted before hard_negative
        if args.mode == "hard_positive":
            data = [d for d in data if 'object' in d]
            random.shuffle(data)
    
    # test run, comment it when you are ready
    # import random
    # random.seed(3)
    # data = random.sample(data, 20)
    # breakpoint()

    sagemaker_runtime = setup()
    init_time = time.time()

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        
        prompt_name, prepared = prepare_input(prompts, datum['prepared'], datum['prediction'], datum['action'], datum['object'], datum['utt'])

        if idx == 0:
            print(prepared)
            # breakpoint()
        if (args.mode in datum):
            continue

        if args.model_name == 'mpt-30b-chat':
            parameters = {'max_new_tokens': args.max_token, "temperature": [args.temperature]}
        elif args.model_name == "falcon-40b-instruct":
            parameters = {"max_new_tokens": args.max_token, "temperature": args.temperature + 0.01}
        else:
            parameters = {"max_new_tokens": args.max_token, "temperature": args.temperature}
        
        messages = {"inputs": prepared, "parameters": parameters}
        if time.time() - init_time > 10 * 60:
            sagemaker_runtime = setup()
            init_time = time.time()
        # sagemaker_runtime = setup()
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=MODEL2ENDPOINTS[args.model_name],
            ContentType='application/json',
            Body=json.dumps(messages).encode('utf-8')
        )

        content, results = parse_completion(response, args.model_name, prepared)
        data[idx][args.mode] = {"results": results, "prompt":prompt_name}

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
    parser.add_argument("--data_path", type=str, required=True)
    # llm
    parser.add_argument("--model_name", type=str, default="mpt-30b-chat")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max_token", type=int, default=512)
    # other
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", type=str, default="hard_positive")
    args = parser.parse_args()

    main(args)