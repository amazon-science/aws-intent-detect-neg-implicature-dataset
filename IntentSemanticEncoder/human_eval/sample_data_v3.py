"""
This version includes split
"""
import random
import logging
import numpy as np
import os, argparse, csv, math
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

    final_examples = []
    final_types = []
    for dataset, negate_version, explanations in zip(["BANKING77", "HWU64", "CLINC150"], ["v4_2", "v1_2", "v1"], ["explanations_for_human_eval", "explanations", "explanations"]):
        args.orig_data_path = f"datasets/{dataset}/test"
        args.imp_data_path_seqin = f"build_toolkit/results/{dataset}/implicature/utterances_gpt-4-0613_in_context_implicature.in"
        args.imp_data_path_label = f"build_toolkit/results/{dataset}/implicature/utterances_gpt-4-0613_in_context_implicature_label"
        # args.neg_data_path_seqin = f"build_toolkit/results/{dataset}/negation/modified_utterance_gpt-3.5-turbo-0613_negate_intent_{dataset}_{negate_version}.in"
        args.neg_data_path_seqin = f"build_toolkit/results/{dataset}/negation/modified_utterance_gpt-4-0613_negate_intent_all_v2.in"
        args.neg_data_path_label = f"datasets/{dataset}/test/label"
        args.candidate_path = f"datasets/{dataset}/{explanations}"

        logger.info("Load data.")
        orig_texts, orig_labels = load_intent_examples(file_path=args.orig_data_path)
        imp_texts, imp_labels = load_intent_examples(seq_file_path=args.imp_data_path_seqin, label_file_path=args.imp_data_path_label)
        neg_texts, neg_labels = load_intent_examples(seq_file_path=args.neg_data_path_seqin, label_file_path=args.neg_data_path_label)
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
                # sample subset of labels
                # sample 1 for each label
                current_examples = []
                for lab in random.sample(list(lab2txt.keys()), args.budget):
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
        
        # enrich the example
        for idx in range(len(examples)):
            lab = examples[idx][1]
            # the order of types and final_examples must be correct and aligned
            final_examples.append([f"{dataset[0]}{idx:05}_0", examples[idx][0], candidates[lab], "", ""])
            final_types.append([f"{dataset[0]}{idx:05}_0", examples[idx][2]])
    
    logger.info("Saving ...")
    order = np.arange(len(final_examples))
    np.random.shuffle(order)
    for anno_id in range(3):
        for idx in range(len(final_examples)):
            final_examples[idx][0] = final_examples[idx][0].replace(f"_{anno_id}", f"_{anno_id+1}")
            final_types[idx][0] = final_types[idx][0].replace(f"_{anno_id}", f"_{anno_id+1}")
        final_examples_ordered = [final_examples[idx] for idx in order]
        final_types_ordered = [final_types[idx] for idx in order]
        output_path = f"human_eval/results/{args.dataset_name}/annotation_{anno_id+1}"
        os.makedirs(output_path, exist_ok=True)
        num_splits = math.ceil(len(final_examples) / args.split_size)
        for i_split in range(num_splits):
            os.makedirs(os.path.join(output_path, str(i_split+1)), exist_ok=True)
            with open(os.path.join(output_path, str(i_split+1), f"annotation_{anno_id+1}_{i_split+1}.csv"), 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["index", "utterance", "intent", "Can the utterance imply the intent?", "Is it conveyed explicitly?"])
                csv_writer.writerows(final_examples_ordered[i_split * args.split_size : (i_split+1) * args.split_size])
            with open(os.path.join(output_path, str(i_split+1), "types.txt"), 'w') as f:
                f.writelines([','.join(t) + "\n" for t in final_types_ordered[i_split * args.split_size : (i_split+1) * args.split_size]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget", type=int, default=20,
                        help="the total number will be 9 * budget.")
    parser.add_argument("--dataset_name", type=str, default="combined")
    parser.add_argument("--split_size", type=int, default=60,
                        help="how many annotations you want in each split")
    args = parser.parse_args()

    main(args)