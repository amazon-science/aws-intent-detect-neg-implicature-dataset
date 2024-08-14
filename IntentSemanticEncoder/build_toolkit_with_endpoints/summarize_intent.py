# some of the object can not be found
# we use LLM to summarize
import os, argparse, json, string, time
from tqdm import tqdm
from endpoints_tools import setup, MODEL2ENDPOINTS

def prepare_input(prompt: str, previous_prompt: str, previous_pred: str, action: str):
    prepared = prompt.replace("[PREVIOUS]", previous_prompt)
    prepared = prepared.replace("[PRED]", previous_pred)
    prepared = prepared.replace("[ACTION]", action)
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
            results = output[prompt_len:]
            # period_ind = results.index(".")
            # results = results[:period_ind+1]
            results = results.strip().lower()
            results = results.split("\n")[0]
            results = results.translate(str.maketrans('', '', string.punctuation))
        else:
            raise NotImplementedError(f"Model {model_name} not included.")
    except Exception as e:
        # you will have to parse it manually here
        print(e)
        # print(completion)
        print(output)
        results = ""
    return output, results

def main(args):
    with open(args.prompt_path, 'r') as f:
        prompt = ''.join(f.readlines())
    
    # get file name of prompt
    prompt_name = os.path.splitext(args.prompt_path)[0].split("/")[-1]

    os.makedirs(f"build_toolkit_with_endpoints/results/pretrain", exist_ok=True)
    assert args.data_path.endswith("_parsed.json")
    pred_path = args.data_path.replace("_parsed.json", f"_{prompt_name}.json")
    if os.path.exists(pred_path) and not args.overwrite:
        # I like to save result data into another file
        # so in the end, we have to open both of them
        # and see which predicted one hasn't been parsed yet.
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        for d, pd in zip(data, pred_data):
            if 'object' in pd:
                d['object'] = pd['object']
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    
    # test run, comment it when you are ready
    # import random
    # random.seed(3)
    # data = random.sample(data[:10000], 20)
    # breakpoint()

    sagemaker_runtime = setup()
    init_time = time.time()

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        # save intermediate results to avoid any termination of program
        if idx % args.save_every == 0 and idx > 0:
            print(f"Save data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                json.dump(data, f, indent=4)
        
        if ('object' in datum) or ('action' not in datum) or ('prediction' not in datum) or (datum['prediction'] == ''):
            continue
        
        prepared = prepare_input(prompt, datum['prepared'], datum['prediction'], datum['action'])

        if idx == 0:
            print(prepared)
            # breakpoint()

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
        # data[idx]['content'] = content
        data[idx]['object'] = results
        # breakpoint()
    
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