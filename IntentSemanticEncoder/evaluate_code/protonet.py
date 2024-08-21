"""
Evaluate encoders with few-/zero- shot classification
"""
import os, argparse, json
import h5py
import torch
import logging
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

def measure(e1, e2, metric):
    if metric == "cosine":
        e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
        e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)
        prod = 1 - e1 @ e2.T # (n, c)
        rank = np.argsort(prod, axis=-1)
        return rank
    else:
        raise NotImplementedError(f"{metric} is not implemented!")

def main(args):
    # create dir
    save_dir = '/'.join(args.save_log_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)

    # encoding
    if args.save_emb_path and os.path.exists(args.save_emb_path) and not args.overwrite:
        # load data
        logger.info(f"Load test intent data.")
        _, lab = load_intent_examples(file_path=args.test_data_path, seq_file_path=args.test_data_path_seqin, label_file_path=args.test_data_path_label)
        logger.info(f"Load intent candidates from {args.candidate_path}.")
        candidates = load_label_file(args.candidate_path)
        if args.num_shots != 0:
            logger.info(f"Load train intent data from {args.train_data_path}")
            _, train_lab = load_intent_examples(args.train_data_path)
        # restore embeddings
        logger.info(f"Restore from {args.save_emb_path} ...")
        with h5py.File(args.save_emb_path, 'r') as f:
            utt_emb = np.asarray(f['utt_emb'])
            lab_emb = np.asarray(f['lab_emb'])
            if args.num_shots != 0:
                train_utt_emb = np.asarray(f['train_utt_emb'])
        lab_txt = [candidates[k] for k in sorted(list(candidates.keys()))]
        # breakpoint()
        lab_idx = [lab_txt.index(candidates[l]) for l in lab]
    else:
        # load data
        logger.info(f"Load test intent data.")
        utt, lab = load_intent_examples(file_path=args.test_data_path, seq_file_path=args.test_data_path_seqin, label_file_path=args.test_data_path_label)
        logger.info(f"Load intent candidates from {args.candidate_path}.")
        candidates = load_label_file(args.candidate_path)
        if args.num_shots != 0:
            logger.info(f"Load train intent data from {args.train_data_path}")
            train_utt, train_lab = load_intent_examples(args.train_data_path)
        assert len(utt) == len(lab)
        lab_txt = [candidates[k] for k in sorted(list(candidates.keys()))]
        lab_idx = [lab_txt.index(candidates[l]) for l in lab]

        # check everything is ok before encoding
        logger.info(f"Labels: {lab_txt}")
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
            utt = [[prompt, u] for u in utt]
            # TODO: should we change prompt for label?
            lab_txt = [[prompt, l] for l in lab_txt]
            if args.num_shots != 0:
                train_utt = [[prompt, u] for u in train_utt]
        else:
            raise NotImplementedError()
        
        utt_emb = model.encode(utt, device="cuda")
        lab_emb = model.encode(lab_txt, device="cuda")
        if args.num_shots != 0:
            train_utt_emb = model.encode(train_utt, device="cuda")

        if args.save_emb_path:
            logger.info(f"Save embeddings to {args.save_emb_path}.")
            with h5py.File(args.save_emb_path, 'w') as f:
                f.create_dataset('utt_emb', data=utt_emb)
                f.create_dataset('lab_emb', data=lab_emb)
                if args.num_shots != 0:
                    f.create_dataset('train_utt_emb', data=train_utt_emb)
            logger.info("Finish")
    
    # evaluation
    if isinstance(lab_txt[0], list) and len(lab_txt[0]) > 1:
        lab_txt = [l[1] for l in lab_txt]
    if args.num_shots != 0:
        lab_emb_dict = defaultdict(list)
        for l, emb in zip(train_lab, train_utt_emb):
            lab_emb_dict[l].append(emb)
        for l in sorted(list(candidates.keys())):
            emb = lab_emb[lab_txt.index(candidates[l])]
            lab_emb_dict[l].append(emb)
        # replace lab_emb with the averaged embeddings
        for l in sorted(list(candidates.keys())):
            lab_emb[lab_txt.index(candidates[l]), :] = np.mean(np.stack(lab_emb_dict[l], 0), 0)
    rank = measure(utt_emb, lab_emb, metric=args.metric)
    assert len(rank) == len(utt_emb)
    assert rank.shape[1] == len(lab_emb)
    accuracy = accuracy_score(lab_idx, rank[:, 0])
    logger.info(f"Accuracy: {accuracy}")

    # log
    # save rank, accuracy
    logger.info(f"Save logs to {args.save_log_path}.")
    with open(args.save_log_path, 'w') as f:
        json.dump({
            "rank": [[int(idx) for idx in r] for r in rank],
            "accuracy": float(accuracy)
        }, f)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    # still have to provide data because label is not saved
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_data_path_seqin", type=str, default=None)
    parser.add_argument("--test_data_path_label", type=str, default=None)
    parser.add_argument("--candidate_path", type=str, required=True)
    parser.add_argument("--save_emb_path", type=str, default=None)
    parser.add_argument("--save_log_path", type=str, required=True)
    # encoder
    parser.add_argument("--encoder_name", type=str, default="hkunlp/instructor-large")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    # other
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--metric", type=str, default="cosine",
                        help="cosine|euclidean")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite embeddings")
    args = parser.parse_args()

    main(args)