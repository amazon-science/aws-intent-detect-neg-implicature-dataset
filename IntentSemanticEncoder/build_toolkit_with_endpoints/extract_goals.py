"""
Goal Extraction is a preprocessing step for pre-train data generation.
It takes as input an utterance and output user goals.
"""
import os, argparse, json, time
from tqdm import tqdm
from endpoints_tools import setup, MODEL2ENDPOINTS

def prepare_input(prompt: str, utterance: str):
    prepared = prompt.replace("[UTTERANCE]", utterance)
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
            # TODO: sometimes there are no periods, this is problematic.
            # could we change to find line breaker?
            period_ind = results.index(".")
            results = results[:period_ind+1]
            results = results.strip()
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
    pred_path = os.path.join(f"build_toolkit_with_endpoints/results/pretrain/extracted_intents_{args.model_name}_{prompt_name}.json")
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            data = [{"utt": l.strip()} for l in f.readlines()]
        for datum in data:
            datum['prepared'] = prepare_input(prompt, datum['utt'])
    
    # test run, comment it when you are ready
    # import random
    # random.seed(3)
    # data = random.sample(data, 20)

    sagemaker_runtime = setup()
    init_time = time.time()

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print(datum['prepared'])
            # breakpoint()
        if 'prediction' in datum:
            continue

        if args.model_name == 'mpt-30b-chat':
            parameters = {'max_new_tokens': args.max_token, "temperature": [args.temperature]}
        elif args.model_name == "falcon-40b-instruct":
            parameters = {"max_new_tokens": args.max_token, "temperature": args.temperature + 0.01}
        else:
            parameters = {"max_new_tokens": args.max_token, "temperature": args.temperature}
        
        messages = {"inputs": datum['prepared'], "parameters": parameters}
        if time.time() - init_time > 10 * 60:
            sagemaker_runtime = setup()
            init_time = time.time()
        # sagemaker_runtime = setup()
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=MODEL2ENDPOINTS[args.model_name],
            ContentType='application/json',
            Body=json.dumps(messages).encode('utf-8')
        )

        content, results = parse_completion(response, args.model_name, datum['prepared'])
        # data[idx]['content'] = content
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