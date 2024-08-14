"""
retrieve:
* random positives
* negatives that have similar intent and close distance

TODO:
1. what if there are no positives?
2. are we able to avoid encoding repeatly?
3. make sure to use the same model for retrieval and pre-training
"""

import torch
import numpy as np
from tqdm import tqdm
import json, os, argparse, random
# import scipy.spatial.distance as distance
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

# def measure(e1, e2, metric):
#     # use for-loop, not a smart way but should be correct
#     if metric == "cosine":
#         return np.asarray([distance.cosine(e1, i2) for i2 in e2])
#     elif metric == "euclidean":
#         return np.asarray([distance.euclidean(e1, i2) for i2 in e2])
#     else:
#         raise NotImplementedError(f"{metric} is not implemented!")

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    if 'instructor' in args.encoder_name:
        model = INSTRUCTOR(args.encoder_name)
        prompt = "Represent the purpose for retrieval: "
    else:
        model = SentenceTransformer(args.encoder_name)
        prompt = None

    assert args.data_path.endswith(".json")
    pred_path = args.data_path.replace(".json", f"_ret_{args.encoder_name.split('/')[-1]}.json")
    print(pred_path)
    if os.path.exists(pred_path) and not args.overwrite:
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    
    # intents as a separate list in the same order
    intents = [d['action'] + d['object'] for d in data]

    # https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
    measure = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    # comment this when you are ready
    # for idx, datum in tqdm(enumerate(data[:200]), total=len(data)):
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        # a initial run shows that 189 / 200 utterances can have retrieved positive
        # this might be enough
        if 'retrieved_positive' not in datum:
            # sample positive
            positive_idx = [i for i in range(len(intents)) if intents[i] == intents[idx] and i != idx]
            if len(positive_idx) > 0:
                # these are all unique utterances, there is no need to check if they are equal
                datum['retrieved_positive'] = [data[random.choice(positive_idx)]['utt']]
        if 'retrieved_negative' not in datum:
            # sample negative
            positive_idx = [i for i in range(len(intents)) if intents[i] == intents[idx]] # notice how this includes itself, so that later you can remove itself
            hard_negative_idx = [i for i in range(len(data)) if data[i]['action'] == data[idx]['action'] or data[i]['object'] == data[idx]['object']]
            hard_negative_idx = list(set(hard_negative_idx) - set(positive_idx))
            # breakpoint()

            if 0 < len(hard_negative_idx) <= 15:
                # less than half batch
                datum['retrieved_negative'] = [data[random.choice(hard_negative_idx)]['utt']]
            elif len(hard_negative_idx) > 15:
                # default batch_size is 32: https://github.com/xlang-ai/instructor-embedding/blob/0658005fa9769401f869b227f7d2cb0ebc933ebd/InstructorEmbedding/instructor.py#L479C37-L479C37
                # so this might be better a multiple of 32
                prepared = [datum['utt']] + [data[i]['utt'] for i in random.sample(hard_negative_idx, min(255, len(hard_negative_idx)))]
                if prompt is not None:
                    prepared = [[prompt, txt] for txt in prepared]

                # the output will be tensor on cuda
                embeds = model.encode(prepared, convert_to_tensor=True, device="cuda")
                hn_embeds = embeds[1:, :]
                # (1, D) and (255, D)
                dists = 1 - measure(embeds[:1, :], hn_embeds) # smaller closer
                dists = dists.cpu().numpy()
                # ranked = [prepared[i+1] for i in np.argsort(dists)]
                # breakpoint()
                # empirically observe that the middle one might be a better choice
                # if it is too close, then it might be a false negative
                # if it is too far, then it might be an easy negative
                if isinstance(prepared[0], list):
                    datum['retrieved_negative'] = [prepared[np.argsort(dists)[len(dists)//2] + 1][1]] # small to large
                else:
                    # a bug for st without prompt
                    datum['retrieved_negative'] = [prepared[np.argsort(dists)[len(dists)//2] + 1]] # small to large
        
        # save intermediate results to avoid any termination of program
        if idx % args.save_every == 0 and idx > 0:
            print(f"Save data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                json.dump(data, f, indent=4)
    
    with open(pred_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--data_path", type=str, required=True)
    # encoder
    parser.add_argument("--encoder_name", type=str, default="hkunlp/instructor-large")
    # other
    parser.add_argument("--metric", type=str, default="cosine",
                        help="cosine|euclidean")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite embeddings")
    args = parser.parse_args()

    main(args)