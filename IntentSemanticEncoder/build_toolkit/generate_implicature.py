"""
Generate implicature for a scenario.
"""
import openai
import os, argparse, json
from tqdm import tqdm
from openai_tools import delayed_completion

def prepare_input(prompt: str, intent: str, scenario: str, num: int, domain: str):
    prepared = prompt.replace("[SCENARIO]", scenario)
    prepared = prepared.replace("[NUM]", str(num))
    prepared = prepared.replace("[DOMAIN]", domain)
    prepared = prepared.replace("[INTENT]", intent)
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
            assert f"utterance_{i+1}" in results
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
    pred_path = os.path.join(f"build_toolkit/results/{args.dataset}/implicature/utterances_{args.model_name}_{prompt_name}.json")
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        assert os.path.exists(args.scenario_path)
        with open(args.scenario_path, 'r') as f:
            brain_storm_results = json.load(f)
        data = []
        for item in brain_storm_results:
            scenarios = [item['prediction'][f'scenario_{i+1}'] for i in range(len(item['prediction']))]
            prepared = []
            for sno in scenarios:
                if isinstance(sno, dict):
                    if "message" in sno:
                        del sno["message"]
                    sno = json.dumps(sno, indent=4)
                assert isinstance(sno, str)
                prepared.append(prepare_input(prompt, item['intent'], sno, args.num, item['domain']))
            scenarios = [{
                "scenario": sno,
                "intent": item['intent'],
                "label": item['label'],
                "domain": item['domain'],
                "prepared": prep} for sno, prep in zip(scenarios, prepared)
            ]
            data.extend(scenarios)
    
    # test run, comment it when you are ready
    # import random
    # random.seed(3)
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
    
    # just for easy to use
    all_utterances = []
    all_intents = []
    for datum in data:
        for utt in datum['prediction']:
            all_utterances.append(datum['prediction'][utt]+"\n")
            all_intents.append(datum['label']+"\n")

    with open(pred_path.replace(".json", ".in"), 'w') as f:
        f.writelines(all_utterances)
    with open(pred_path.replace(".json", "_label"), 'w') as f:
        f.writelines(all_intents)
    
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--scenario_path", type=str, required=True)
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
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)