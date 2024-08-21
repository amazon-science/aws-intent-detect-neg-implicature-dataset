"""
Evaluate encoders with triplet task
"""
import os, argparse, json
import h5py
import torch
import random
import logging
import numpy as np
from collections import defaultdict
import scipy.spatial.distance as distance

from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_intent_examples(file_path=None, seq_file_path=None, label_file_path=None):
    texts, labels = [], []
    # text files are organized so that texts and labels are saved
    # in separate files. Use the line index to correspond two.
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

def load_utterances(file_path):
    texts = []
    with open(file_path, 'r') as f:
        texts = f.readlines()
    texts = [txt.strip() for txt in texts]
    return texts

def measure(e1, e2, metric):
    # use for-loop, not a smart way but should be correct
    if metric == "cosine":
        return np.asarray([distance.cosine(i1, i2) for i1, i2 in zip(e1, e2)])
    elif metric == "euclidean":
        return np.asarray([distance.euclidean(i1, i2) for i1, i2 in zip(e1, e2)])
    else:
        raise NotImplementedError(f"{metric} is not implemented!")

def main(args):
    # set seed
    logger.info(f"Set seed to {args.seed}.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create dir
    save_dir = '/'.join(args.save_log_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)

    # encoding
    if args.save_emb_path and os.path.exists(args.save_emb_path) and not args.overwrite:
        logger.info(f"Restore from {args.save_emb_path} ...")
        with h5py.File(args.save_emb_path, 'r') as f:
            orig_emb = np.asarray(f['orig_emb'])
            neg_emb = np.asarray(f['neg_emb'])
            pos_emb = np.asarray(f['pos_emb'])
    else:
        # load data
        logger.info(f"Load original data from {args.original_data_path}.")
        orig_utt, orig_lab = load_intent_examples(args.original_data_path)
        logger.info(f"Load negative data from {args.negative_data_path}.")
        neg_utt = load_utterances(args.negative_data_path)
        if args.positive_data_path or args.positive_data_seqin or \
            args.positive_data_label:
            logger.info(f"Load positive data.")
            pos_utt, pos_lab = load_intent_examples(args.positive_data_path, args.positive_data_seqin, args.positive_data_label)

            if args.sample_positive:
                logger.info("Sample positive data.")
                same_lab_dict = defaultdict(list)
                for utt, lab in zip(pos_utt, pos_lab):
                    same_lab_dict[lab].append(utt)
                pos_utt = []
                for utt, lab in zip(orig_utt, orig_lab):
                    rnd_utt = random.choice(same_lab_dict[lab])
                    # make sure that the positive one is different from original one
                    while rnd_utt == utt:
                        rnd_utt = random.choice(same_lab_dict[lab])
                    pos_utt.append(rnd_utt)
            else:
                # if do not sample positive, then make sure that positive labels align with original labels
                for ol, pl in zip(orig_lab, pos_lab):
                    assert ol == pl
        else:
            logger.info("Sample positive data from original data.")
            same_lab_dict = defaultdict(list)
            for utt, lab in zip(orig_utt, orig_lab):
                same_lab_dict[lab].append(utt)
            pos_utt = []
            for utt, lab in zip(orig_utt, orig_lab):
                rnd_utt = random.choice(same_lab_dict[lab])
                # make sure that the positive one is different from original one
                while rnd_utt == utt:
                    rnd_utt = random.choice(same_lab_dict[lab])
                pos_utt.append(rnd_utt)
        assert len(orig_utt) == len(neg_utt) == len(pos_utt) == len(orig_lab)

        # check everything is ok before encoding
        # breakpoint()

        logger.info("Encoding utterances ...")
        if args.encoder_name == "iae" and args.model_name_or_path:
            model = SentenceTransformer(args.model_name_or_path)
        elif 'sentence-transformers' in args.encoder_name:
            # one of st models: https://huggingface.co/sentence-transformers
            model = SentenceTransformer(args.encoder_name)
            if args.model_name_or_path is not None:
                logger.info(f"Loading from {args.model_name_or_path} ...")
                if os.path.exists(os.path.join(args.model_name_or_path, 'config.json')):
                    model = SentenceTransformer(args.model_name_or_path)
                else:
                    state_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'))
                    model.load_state_dict(state_dict)
        elif "instructor" in args.encoder_name:
            model = INSTRUCTOR(args.encoder_name)
            if args.model_name_or_path is not None:
                logger.info(f"Loading from {args.model_name_or_path} ...")
                if os.path.exists(os.path.join(args.model_name_or_path, 'config.json')):
                    model = INSTRUCTOR(args.model_name_or_path)
                else:
                    state_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'))
                    model.load_state_dict(state_dict)
            # https://github.com/HKUNLP/instructor-embedding/blob/548ac97db1f79e284fe237282789a98cbba9861a/evaluation/MTEB/mteb/evaluation/evaluators/ClassificationEvaluator.py#L17
            # https://github.com/HKUNLP/instructor-embedding/blob/548ac97db1f79e284fe237282789a98cbba9861a/evaluation/MTEB/mteb/evaluation/evaluators/ClusteringEvaluator.py#L46C84-L46C84
            prompt = "Represent the purpose for retrieval: "
            # https://github.com/HKUNLP/instructor-embedding/tree/main
            orig_utt = [[prompt, utt] for utt in orig_utt]
            neg_utt = [[prompt, utt] for utt in neg_utt]
            pos_utt = [[prompt, utt] for utt in pos_utt]
        else:
            raise NotImplementedError()
        
        orig_emb = model.encode(orig_utt, device="cuda")
        neg_emb = model.encode(neg_utt, device="cuda")
        pos_emb = model.encode(pos_utt, device="cuda")

        if args.save_emb_path:
            logger.info(f"Save embeddings to {args.save_emb_path}.")
            with h5py.File(args.save_emb_path, 'w') as f:
                f.create_dataset('orig_emb', data=orig_emb)
                f.create_dataset('neg_emb', data=neg_emb)
                f.create_dataset('pos_emb', data=pos_emb)
            logger.info("Finish")
    
    # evaluation
    # task 1
    d_on = measure(orig_emb, neg_emb, args.metric)
    d_op = measure(orig_emb, pos_emb, args.metric)
    # depends on the metric but will be handled in measure function
    success_t1 = d_on > d_op
    accuracy_t1 = success_t1.sum() / len(success_t1)
    logger.info(f"Task 1 accuracy: {accuracy_t1}")
    # task 2
    d_np = measure(neg_emb, pos_emb, args.metric)
    # depends on the metric but will be handled in measure function
    success_t2 = d_np > d_op
    accuracy_t2 = success_t2.sum() / len(success_t2)
    logger.info(f"Task 2 accuracy: {accuracy_t2}")

    # log
    # save distance, success and accuracy
    logger.info(f"Save logs to {args.save_log_path}.")
    with open(args.save_log_path, 'w') as f:
        json.dump({
            "distance_orig_neg": [float(item) for item in d_on],
            "distance_orig_pos": [float(item) for item in d_op],
            "distance_neg_pos": [float(item) for item in d_np],
            "success_t1": [bool(item) for item in success_t1],
            "success_t2": [bool(item) for item in success_t2],
            "accuracy_t1": float(accuracy_t1),
            "accuracy_t2": float(accuracy_t2)
        }, f)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    # there is no need to provide data if you already saved embeddings
    parser.add_argument("--original_data_path", type=str, default=None)
    parser.add_argument("--negative_data_path", type=str, default=None)
    parser.add_argument("--positive_data_path", type=str, default=None)
    parser.add_argument("--positive_data_seqin", type=str, default=None)
    parser.add_argument("--positive_data_label", type=str, default=None)
    parser.add_argument("--save_emb_path", type=str, default=None)
    parser.add_argument("--save_log_path", type=str, required=True)
    # encoder
    parser.add_argument("--encoder_name", type=str, default="hkunlp/instructor-large")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    # other
    parser.add_argument("--metric", type=str, default="cosine",
                        help="cosine|euclidean")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite embeddings")
    parser.add_argument("--sample_positive", action="store_true",
                        help="sample from positive data")
    args = parser.parse_args()

    main(args)