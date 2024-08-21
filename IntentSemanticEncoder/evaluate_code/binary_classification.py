"""
Evaluate encoders with binary classification task (currently only support zero-shot setting)
the class names are one positive and one negative
"""
import os, argparse, json
import h5py
import torch
import logging
import numpy as np
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
        e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True) # (n, d)
        e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True) # (n, 2, d)
        prod = 1 - e1[:, None, :] @ np.transpose(e2, (0, 2, 1)) # (n, 1, 2)
        rank = np.argsort(prod.squeeze(), axis=-1)
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
        _, lab = load_intent_examples(file_path=args.data_path, seq_file_path=args.data_path_seqin, label_file_path=args.data_path_label)
        logger.info(f"Load intent candidates from {args.candidate_path}.")
        candidates = load_label_file(args.candidate_path)
        logger.info(f"Load negative intent candidates from {args.neg_candidate_path}.")
        neg_candidates = load_label_file(args.neg_candidate_path)
        # restore embeddings
        logger.info(f"Restore from {args.save_emb_path} ...")
        with h5py.File(args.save_emb_path, 'r') as f:
            utt_emb = np.asarray(f['utt_emb'])
            lab_emb = np.asarray(f['lab_emb'])
        # lab_txt = [[candidates[k], neg_candidates[k]] for k in sorted(list(candidates.keys()))]
        # breakpoint()
    else:
        # load data
        logger.info(f"Load test intent data.")
        utt, lab = load_intent_examples(file_path=args.data_path, seq_file_path=args.data_path_seqin, label_file_path=args.data_path_label)
        logger.info(f"Load intent candidates from {args.candidate_path}.")
        candidates = load_label_file(args.candidate_path)
        logger.info(f"Load negative intent candidates from {args.neg_candidate_path}.")
        neg_candidates = load_label_file(args.neg_candidate_path)
        assert len(utt) == len(lab)
        lab_txt = [[candidates[k], neg_candidates[k]] for k in sorted(list(candidates.keys()))]

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
        else:
            raise NotImplementedError()

        lab_txt_flat = sum(lab_txt, [])
        if "instructor" in args.encoder_name:
            lab_txt_flat = [[prompt, l] for l in lab_txt_flat]
        # breakpoint()
        utt_emb = model.encode(utt, device="cuda")
        lab_emb = model.encode(lab_txt_flat, device="cuda")
        # re-organize into (c, 2, d)
        lab_emb = np.asarray(np.array_split(lab_emb, len(lab_emb)//2, axis=0))
        assert len(lab_emb) == len(candidates)
        assert lab_emb.shape[1] == 2

        if args.save_emb_path:
            logger.info(f"Save embeddings to {args.save_emb_path}.")
            with h5py.File(args.save_emb_path, 'w') as f:
                f.create_dataset('utt_emb', data=utt_emb)
                f.create_dataset('lab_emb', data=lab_emb)
            logger.info("Finish")
    
    # evaluation
    lab_emb_dict = {k:emb for k, emb in zip(sorted(list(candidates.keys())), lab_emb)}
    lab_emb_all = [lab_emb_dict[l] for l in lab]
    rank = measure(utt_emb, lab_emb_all, metric=args.metric)
    assert len(rank) == len(utt_emb)
    assert rank.shape[1] == 2
    # !warning: always choose positive
    accuracy = accuracy_score([0] * len(lab), rank[:, 0])
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
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_path_seqin", type=str, default=None)
    parser.add_argument("--data_path_label", type=str, default=None)
    parser.add_argument("--candidate_path", type=str, required=True)
    parser.add_argument("--neg_candidate_path", type=str, required=True)
    parser.add_argument("--save_emb_path", type=str, default=None)
    parser.add_argument("--save_log_path", type=str, required=True)
    # encoder
    parser.add_argument("--encoder_name", type=str, default="hkunlp/instructor-large")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    # other
    parser.add_argument("--metric", type=str, default="cosine",
                        help="cosine|euclidean")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite embeddings")
    args = parser.parse_args()

    main(args)