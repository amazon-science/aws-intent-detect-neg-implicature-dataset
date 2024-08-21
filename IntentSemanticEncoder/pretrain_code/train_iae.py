# Deprecated!
"""
This code runs IAE model for instructor.
It is used as a baseline to compare with our method.

Things to change:
* model forward
* tokenization
* add prompt

ISSUE: it is very hard to adapt instructor model to this code. there
are a lot of details you need to consider, such as:
1. selection of hyperparameters
2. adding heads for contrastive learning
3. correct tokenization
Even it was correctly implemented, it is still a question whether this is
comparable to the original implementation.
"""

import argparse
import copy
import logging
import os
import time
import shutil
import torch
# from eval_iae import run_eval
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from iae.contrastive_learning import ContrastiveLearningPairwise
from iae.data_loader import ContrastiveLearningDataset
from iae.drophead import set_drophead
# from iae.iae_model import IAEModel
from iae.utils import init_logging
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer

LOGGER = logging.getLogger()

def train(args, data_loader, model, scaler=None, step_global=0):
    LOGGER.info("train!")
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    # best_acc = 0
    # best_model = copy.deepcopy(iae_model)
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # breakpoint()
        model.optimizer.zero_grad()
        # batch_x1: input utterance
        # batch_x2: gold intent
        # batch_x3: gold utterance
        # batch_x4: pseudo intent

        batch_x1, batch_x2, batch_x3, batch_x4 = copy.deepcopy(data)
        
        batch_x_cuda1, batch_x_cuda2, batch_x_cuda3, batch_x_cuda4 = {},{},{},{}
        for k,v in batch_x1.items():
            batch_x_cuda1[k] = torch.LongTensor(v).cuda()
        for k,v in batch_x2.items():
            batch_x_cuda2[k] = torch.LongTensor(v).cuda()
        for k,v in batch_x3.items():
            batch_x_cuda3[k] = torch.LongTensor(v).cuda()
        for k,v in batch_x4.items():
            batch_x_cuda4[k] = torch.LongTensor(v).cuda()
        encoded1 = model.encoder(batch_x_cuda1)['sentence_embedding']
        encoded2 = model.encoder(batch_x_cuda2)['sentence_embedding']
        encoded3 = model.encoder(batch_x_cuda3)['sentence_embedding']
        encoded4 = model.encoder(batch_x_cuda4)['sentence_embedding']

        if args.amp:
            with autocast():
                loss = torch.tensor(0.0, requires_grad=True).cuda()
                loss += model(encoded1, encoded2)
                loss += model(encoded1, encoded3)
                if args.pseudo_weight: loss += args.pseudo_weight * model(encoded1, encoded4)
        else:
            loss = torch.tensor(0.0, requires_grad=True).cuda()
            loss += model(encoded1, encoded2)
            loss += model(encoded1, encoded3)
            if args.pseudo_weight: loss += args.pseudo_weight * model(encoded1, encoded4)
        
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        if torch.isnan(loss):
            print("Loss is nan")
            print(data)
            break
        train_steps += 1
        step_global += 1

        # if args.eval_during_training and (step_global % args.eval_step == 0):
        #     checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_tmp")
        #     if not os.path.exists(checkpoint_dir):
        #         os.makedirs(checkpoint_dir)
        #     iae_model.save_model(checkpoint_dir)
            
        #     acc = run_eval(test_path=args.val_path,
        #              model_name_or_path=checkpoint_dir,
        #              distance_metric=args.distance_metric)

        #     if acc > best_acc:
        #         best_acc = acc
        #         best_model = copy.deepcopy(iae_model)
            
        #     LOGGER.info(f"step:{step_global}, val_acc:{acc}")

    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global
    
def main(args):
    init_logging(LOGGER)
    print(args)

    torch.manual_seed(args.random_seed) 
    # by default 42 is used, also tried 33, 44, 55
    # results don't seem to change too much
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load BERT tokenizer, dense_encoder
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True,
        revision="main",
        use_auth_token=None
    )
    # breakpoint()
    encoder = INSTRUCTOR(args.model_name_or_path, cache_folder=args.cache_dir)

    # adjust dropout rates
    # normally we do not use higher dropout in instructor
    # disable this for fair comparison
    # encoder.embeddings.dropout = torch.nn.Dropout(p=args.dropout_rate)
    # for i in range(len(encoder.encoder.layer)):
        # hotfix
        # try:
        #     encoder.encoder.layer[i].attention.self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        #     encoder.encoder.layer[i].attention.output.dropout = torch.nn.Dropout(p=args.dropout_rate)
        # except:
        #     encoder.encoder.layer[i].attention.attn.dropout = torch.nn.Dropout(p=args.dropout_rate)
        #     encoder.encoder.layer[i].attention.dropout = torch.nn.Dropout(p=args.dropout_rate)

        # encoder.encoder.layer[i].output.dropout  = torch.nn.Dropout(p=args.dropout_rate)

    # set drophead rate
    # if args.drophead_rate != 0:
    #     set_drophead(encoder, args.drophead_rate)

    prompt = "Represent the purpose for retrieval: "
    def collate_fn_batch_encoding(batch):
        # TODO: modify this function because instructor needs context_mask
        # see preprocess_function in instructor training
        # utterance, gold_intent, gold_utterance, pseudo_intent
        sent1, sent2, sent3, sent4 = zip(*batch)
        results = []
        for examples in [sent1, sent2, sent3, sent4]:
            num = len(examples)
            contexts = []
            concatenated_input_texts = []
            for local_idx in range(num):
                splits = [prompt, examples[local_idx]]
                # splits = examples[local_idx].split('!@#$%^&**!@#$%^&**')
                # assert len(splits) == 2
                contexts.append(splits[0])
                concatenated_input_texts.append(''.join(splits))
                assert isinstance(contexts[-1], str)
                assert isinstance(concatenated_input_texts[-1], str)
            tokenized = tokenizer(concatenated_input_texts,padding='max_length', truncation='longest_first', return_tensors="pt", max_length=args.max_length)
            context_tok = tokenizer(contexts,padding='max_length', truncation='longest_first', return_tensors="pt", max_length=args.max_length)
            tokenized['context_masks'] = torch.sum(context_tok['attention_mask'], dim=1)
            tokenized['context_masks'] = tokenized['context_masks'] - 1
            for my_idx in range(len(tokenized['context_masks'])):
                if tokenized['context_masks'][my_idx] <= 1:
                    tokenized['context_masks'][my_idx] = 0
            for k in tokenized.keys():
                tokenized[k] = tokenized[k].tolist()
            results.append(tokenized)
        return results[0], results[1], results[2], results[3]

    train_set = ContrastiveLearningDataset(
        args.train_path,
        tokenizer=tokenizer,
        random_span_mask=args.random_span_mask,
        draft=args.draft
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn_batch_encoding,
        drop_last=True
    )
    # breakpoint()
    # TODO: modify this since instructor has a different forwarding
    model = ContrastiveLearningPairwise(
        encoder=encoder,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        infoNCE_tau=args.infoNCE_tau,
        agg_mode=args.agg_mode
    )
    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("using nn.DataParallel")
    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    # TODO: there is only one epoch training, no need to validate
    step_global = 0
    for epoch in range(1,args.epoch+1):
        LOGGER.info(f"Epoch {epoch}/{args.epoch}")

        # train
        train_loss, step_global = train(args, data_loader=train_loader, model=model, 
                scaler=scaler, step_global=step_global)
        LOGGER.info(f'loss/train_per_epoch={train_loss}/{epoch}')

    # TODO: modify saving function
    model.encoder.save(args.output_dir)
    

    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info(f"Training Time!{training_hour} hours {training_minute} minutes {training_second} seconds")
    
if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train IAE Model')

    # Required
    parser.add_argument('--train_path', type=str, required=True, help='training set directory')
    parser.add_argument('--val_path', type=str, help='validation set directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output')
    parser.add_argument('--cache_dir', type=str, required=True)

    parser.add_argument('--model_name_or_path', type=str, \
        help='Directory for pretrained model', \
        default="roberta-base")
    parser.add_argument('--max_length', default=50, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--train_batch_size', default=200, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--infoNCE_tau', default=0.04, type=float) 
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_std}") 
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--parallel', action="store_true") 
    parser.add_argument('--amp', action="store_true", \
        help="automatic mixed precision training")
    parser.add_argument('--random_seed', default=42, type=int)

    # data augmentation config
    parser.add_argument('--dropout_rate', default=0.1, type=float) 
    parser.add_argument('--drophead_rate', default=0.0, type=float)
    parser.add_argument('--random_span_mask', default=5, type=int, 
            help="number of chars to be randomly masked on one side of the input") 

    parser.add_argument('--distance_metric', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--eval_during_training', action='store_true')
    parser.add_argument('--eval_step', default=200, type=int)
    parser.add_argument('--pseudo_weight', default=2, type=float)
    args = parser.parse_args()

    main(args)