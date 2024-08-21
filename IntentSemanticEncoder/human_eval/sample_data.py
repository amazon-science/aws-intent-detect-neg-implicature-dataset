"""
* We should have 3 annotators for each data so that we can check inter-annotator agreement.
* We should include as many intents as possible.
"""

import random
import logging
import numpy as np
import os, argparse, csv
from collections import defaultdict

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
    labels = {l[0]: l[1].replace("_", " ") if len(l) == 2 else l[0].replace("_", " ").lower() for l in lines}
    return labels

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Load data.")
    orig_texts, orig_labels = load_intent_examples(args.orig_data_path, args.orig_data_path_seqin, args.orig_data_path_label)
    imp_texts, imp_labels = load_intent_examples(args.imp_data_path, args.imp_data_path_seqin, args.imp_data_path_label)
    neg_texts, neg_labels = load_intent_examples(args.neg_data_path, args.neg_data_path_seqin, args.neg_data_path_label)
    logger.info("Load label.")
    candidates = load_label_file(args.candidate_path)
    orig_lab2txt = defaultdict(list)
    for txt, lab in zip(orig_texts, orig_labels):
        orig_lab2txt[lab].append(txt)
    imp_lab2txt = defaultdict(list)
    for txt, lab in zip(imp_texts, imp_labels):
        imp_lab2txt[lab].append(txt)
    neg_lab2txt = defaultdict(list)
    for txt, lab in zip(neg_texts, neg_labels):
        neg_lab2txt[lab].append(txt)
    
    logger.info("Sample examples")
    examples = []
    for lab2txt, exp_type in zip([orig_lab2txt, imp_lab2txt, neg_lab2txt], ['original', 'implicature', 'negation']):
        if len(lab2txt) >= args.budget:
            # more labels than budget
            # sample 1 for each label
            current_examples = []
            for lab in lab2txt:
                rnd_exps = [(txt, lab, exp_type) for txt in random.sample(lab2txt[lab], 1)]
                current_examples.extend(rnd_exps)
                if len(current_examples) >= args.budget:
                    break
            examples += current_examples
        else:
            # more budget than labels
            # sample ns for each label
            ns = [len(s) for s in np.array_split(np.arange(args.budget), len(lab2txt))]
            for num, lab in zip(ns, lab2txt.keys()):
                rnd_exps = [(txt, lab, exp_type) for txt in random.sample(lab2txt[lab], num)]
                examples.extend(rnd_exps)
    
    logger.info("Saving.")
    order = np.arange(len(examples))
    for anno_id in range(3):
        np.random.shuffle(order)
        final_examples = []
        types = []
        for idx in order:
            lab = examples[idx][1]
            # the order of types and final_examples must be correct and aligned
            final_examples.append([f"{idx:05}_{anno_id+1}", examples[idx][0], candidates[lab], "correct intent or not?", "", "is this implicature?", ""])
            types.append(examples[idx][2])
        
        output_path = f"human_eval/results/{args.dataset}/annotator_{anno_id+1}"
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "data.csv"), 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["index", "utterance", "intent", "Question 1", "Answer 1", "Question 2", "Answer 2"])
            csv_writer.writerows(final_examples)
        with open(os.path.join(output_path, "types.txt"), 'w') as f:
            f.writelines([t + "\n" for t in types])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--orig_data_path", type=str, default=None)
    parser.add_argument("--orig_data_path_seqin", type=str, default=None)
    parser.add_argument("--orig_data_path_label", type=str, default=None)
    parser.add_argument("--imp_data_path", type=str, default=None)
    parser.add_argument("--imp_data_path_seqin", type=str, default=None)
    parser.add_argument("--imp_data_path_label", type=str, default=None)
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_data_path_seqin", type=str, default=None)
    parser.add_argument("--neg_data_path_label", type=str, default=None)
    parser.add_argument("--candidate_path", type=str, required=True)
    # other
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    main(args)