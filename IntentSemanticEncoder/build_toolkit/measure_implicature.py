"""
An automatic keyword-based metric.

TODO:
We should try to implement:
1. tf-idf
2. BLEU
3. ROUGE
4. METEOR
5. BertScore
"""

import json, os, argparse
import nltk, evaluate
import logging
import numpy as np
from typing import List, Dict
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

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


def identify_intent_related_words(train_utterances: List[int], train_labels: List[int], top_k: int) -> Dict[str, List[str]]:
    """
    save the results!
    """
    docs = defaultdict(list)
    for txt, lab in zip(train_utterances, train_labels):
        docs[lab].append(txt)
    corpus = []
    corpus_labels = []
    for lab in docs:
        corpus.append(' '.join(docs[lab]))
        corpus_labels.append(lab)
    
    print("Start fitting tf-idf ...")
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
    vectorizer.fit(corpus)
    vocabs = vectorizer.get_feature_names_out()
    features = vectorizer.transform(corpus)
    print("Finished fitting!")

    lab2feat = defaultdict(list)
    for feat, lab in zip(features, corpus_labels):
        feat = feat.toarray().squeeze()
        feat_inds = np.argsort(feat)[::-1][:top_k]
        for ind in feat_inds:
            lab2feat[lab].append(vocabs[ind])
    
    return lab2feat

def main(args):
    # create dir
    save_dir = '/'.join(args.save_log_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)

    train_texts, train_labels = load_intent_examples(file_path=args.train_data_path)
    texts, labels = load_intent_examples(file_path=args.test_data_path, seq_file_path=args.test_data_path_seqin, label_file_path=args.test_data_path_label)
    if args.method == "keyword":
        kw_path = os.path.join(save_dir, f"intent_related_words_{args.top_k}.json")
        if os.path.exists(kw_path):
            logger.info(f"Restore keywords from {kw_path}.")
            with open(kw_path, 'r') as f:
                lab2feat = json.load(f)
        else:
            logger.info("Identifying intent related words ...")
            lab2feat = identify_intent_related_words(train_texts, train_labels, top_k=args.top_k)
            logger.info("Finish!")
            logger.info(f"Save to {kw_path}.")
            with open(kw_path, 'w') as f:
                json.dump(lab2feat, f)

        preprocessor = TfidfVectorizer().build_preprocessor()
        tokenizer = TfidfVectorizer().build_tokenizer()

        toks = [tokenizer(preprocessor(txt)) for txt in texts]
        kw_ratio = []
        for tok, lab in zip(toks, labels):
            kw = lab2feat[lab]
            is_kw = [w in kw for w in tok]
            kw_ratio.append(sum(is_kw) / len(is_kw))
        measure = np.mean(kw_ratio)
    elif args.method in ["BLEU", "ROUGE", "METEOR", "BertScore"]:
        lab2utt = defaultdict(list)
        for txt, lab in zip(train_texts, train_labels):
            lab2utt[lab].append(txt)
        references = [lab2utt[lab] for lab in labels]
        if args.method == "BLEU":
            metric = evaluate.load("bleu")
            results = metric.compute(predictions=texts, references=references)
            measure = results['bleu']
        elif args.method == "ROUGE":
            metric = evaluate.load("rouge")
            measure = metric.compute(predictions=texts, references=references)
        elif args.method == "METEOR":
            metric = evaluate.load("meteor")
            results = metric.compute(predictions=texts, references=references)
            measure = results["meteor"]
        elif args.method == "BertScore":
            metric = evaluate.load("bertscore")
            measure = metric.compute(predictions=texts, references=references, lang="en", device="cuda")
            measure['precision'] = float(np.mean(measure['precision']))
            measure['recall'] = float(np.mean(measure['recall']))
            measure['f1'] = float(np.mean(measure['f1']))
    else:
        raise NotImplementedError(f"Method {args.method} not implemented!")

    logger.info(f"Measure: {measure}")
    # save measure
    with open(args.save_log_path, 'w') as f:
        json.dump(measure, f)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_data_path_seqin", type=str, default=None)
    parser.add_argument("--test_data_path_label", type=str, default=None)
    parser.add_argument("--save_log_path", type=str, required=True)
    # other
    parser.add_argument("--method", type=str, default="keyword")
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    main(args)
