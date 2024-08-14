# modify the objects in the intents potentially be used for generating utterances
import os, argparse, json, string
from tqdm import tqdm
from endpoints_tools import setup, MODEL2ENDPOINTS

def prepare_input(prompt: str, previous_prompt: str, previous_pred: str, action: str, object: str, utt: str):
    prepared = prompt.replace("[PREVIOUS]", previous_prompt)
    prepared = prepared.replace("[PRED]", previous_pred)
    prepared = prepared.replace("[ACTION]", action)
    prepared = prepared.replace("[OBJECT]", object)
    prepared = prepared.replace("[UTTERANCE]", utt)
    return prepared

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
                results = results.split("\n")[0]
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
    except Exception as e:
        # you will have to parse it manually here
        print(e)
        # print(completion)
        print(output)
        out_list = []
    return output, out_list

def main(args):
    with open(args.prompt_path, 'r') as f:
        prompt = ''.join(f.readlines())
    
    # get file name of prompt
    prompt_name = os.path.splitext(args.prompt_path)[0].split("/")[-1]

    os.makedirs(f"build_toolkit_with_endpoints/results/pretrain", exist_ok=True)
    assert args.data_path.endswith(".json")
    pred_path = args.data_path.replace(".json", f"_{args.model_name}_{prompt_name}.json")
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
            if 'modified_intents' in pd and len(pd['modified_intents']) == 3:
                d['modified_intents'] = pd['modified_intents']
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    
    # test run, comment it when you are ready
    import random
    random.seed(4)
    data = random.sample(data[:10000], 20)
    # breakpoint()

    sagemaker_runtime = setup()

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        # save intermediate results to avoid any termination of program
        if idx % args.save_every == 0 and idx > 0:
            print(f"Save data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                json.dump(data, f, indent=4)
        
        prepared = prepare_input(prompt, datum['prepared'], datum['prediction'], datum['action'], datum['object'], datum['utt'])

        if idx == 0:
            print(prepared)
            # breakpoint()
        if "modified_intents" in datum:
            continue

        if args.model_name == 'mpt-30b-chat':
            parameters = {'max_new_tokens': args.max_token, "temperature": [args.temperature]}
        elif args.model_name == "falcon-40b-instruct":
            parameters = {"max_new_tokens": args.max_token, "temperature": args.temperature + 0.01}
        else:
            parameters = {"max_new_tokens": args.max_token, "temperature": args.temperature}

        messages = {"inputs": prepared, "parameters": parameters}
        # sagemaker_runtime = setup()
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=MODEL2ENDPOINTS[args.model_name],
            ContentType='application/json',
            Body=json.dumps(messages).encode('utf-8')
        )

        content, results = parse_completion(response, args.model_name, prepared)
        data[idx]['modified_intents'] = results
    
    with open(pred_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    # llm
    parser.add_argument("--model_name", type=str, default="mpt-30b-chat")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max_token", type=int, default=512)
    # other
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)