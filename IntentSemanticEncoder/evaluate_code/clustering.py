"""
Evaluate encoders with clustering

TODO:
1. K-means
2. agglomerative clustering
"""
import os, argparse, json
import h5py
import torch
import logging
import numpy as np
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering

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

def main(args):
    # create dir
    save_dir = '/'.join(args.save_log_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)

    # encoding
    if args.save_emb_path and os.path.exists(args.save_emb_path) and not args.overwrite:
        # load data
        logger.info(f"Load test intent data")
        _, lab = load_intent_examples(file_path=args.test_data_path, seq_file_path=args.test_data_path_seqin, label_file_path=args.test_data_path_label)
        # restore embeddings
        logger.info(f"Restore from {args.save_emb_path} ...")
        with h5py.File(args.save_emb_path, 'r') as f:
            utt_emb = np.asarray(f['utt_emb'])
        candidates = {k:idx for idx, k in enumerate(sorted(list(set(lab))))}
        lab_idx = [candidates[l] for l in lab]
    else:
        # load data
        logger.info(f"Load test intent data.")
        utt, lab = load_intent_examples(file_path=args.test_data_path, seq_file_path=args.test_data_path_seqin, label_file_path=args.test_data_path_label)
        assert len(utt) == len(lab)
        candidates = {k:idx for idx, k in enumerate(sorted(list(set(lab))))}
        lab_idx = [candidates[l] for l in lab]

        # check everything is ok before encoding
        logger.info(f"Labels: {sorted(list(set(lab)))}")
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

        utt_emb = model.encode(utt, device="cuda")

        if args.save_emb_path:
            logger.info(f"Save embeddings to {args.save_emb_path}.")
            with h5py.File(args.save_emb_path, 'w') as f:
                f.create_dataset('utt_emb', data=utt_emb)
            logger.info("Finish")
    
    # evaluation
    clustering_models = {"kmeans": KMeans(n_clusters=len(candidates), n_init='auto', random_state=100), "agglomerative": AgglomerativeClustering(n_clusters=len(candidates))}
    clustering_scores = {}
    for cm_name in clustering_models.keys():
        preds = clustering_models[cm_name].fit_predict(utt_emb)
        clustering_scores[cm_name] = normalized_mutual_info_score(lab_idx, preds)
    logger.info(clustering_scores)

    # log
    # save rank, accuracy
    logger.info(f"Save logs to {args.save_log_path}.")
    with open(args.save_log_path, 'w') as f:
        json.dump(clustering_scores, f)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    # still have to provide data because label is not saved
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_data_path_seqin", type=str, default=None)
    parser.add_argument("--test_data_path_label", type=str, default=None)
    parser.add_argument("--save_emb_path", type=str, default=None)
    parser.add_argument("--save_log_path", type=str, required=True)
    # encoder
    parser.add_argument("--encoder_name", type=str, default="hkunlp/instructor-large")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    # other
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite embeddings")
    args = parser.parse_args()

    main(args)
