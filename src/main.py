# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

from data_loader import load_data
from train import train_model, eval_model
from model import DialogVAD, DialogVAD_roberta
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
# CONFIG

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()

args.device        = 1
args.MAX_LEN       = 128
args.MAX_NUM_UTTR  = 40
args.batch_size    = 32
args.adam_epsilon  = 1e-8
# args.epochs        = 3
args.num_class     = 2
# args.drop_out      = 0.1
args.test_size     = 0.1
args.d_transformer = 128


args.mode         = 'Uttr' #Context_Hierarchical_emoberta_uttr' #_emoberta_uttr'
args.BASE         = 'BERT'
args.VAD_tokenized_dict = '../VAD_tokenized_dict.json'
args.result_name  = args.mode + '.txt' 





## get vad dict
VAD_Lexicons = pd.read_csv('../data/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt', sep='\t')
VAD_dict = {}
for r in VAD_Lexicons.iterrows():
    VAD_dict[r[1]['Word']] = [r[1]['Valence'], r[1]['Arousal'], r[1]['Dominance']]
args.VAD_dict = VAD_dict


personalities = ['N']


from transformers import AutoTokenizer, AutoModelForSequenceClassification

if args.BASE == 'BERT':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    epoch_list = [4]
    lr_list = [2e-5]
elif args.BASE == 'RoBERTa':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    epoch_list = [4]
    lr_list = [2e-5]

args.lr = lr_list[0]





cnt = 0

seeds =  [0, 1, 13, 41, 42, 123, 456, 321, 999, 1024] # 0

with open(args.result_name, 'w') as f:
    test_acc_total = []
    for personality in personalities:
        args.lr = lr_list[0]#[cnt]
        args.epochs = epoch_list[0]#[cnt]
        cnt += 1
        df = pd.read_csv('../data/Friends_'+personality+'_whole.tsv', sep='\t')
        print('Current training classifier for', personality, '...')

        test_acc_all_seeds = []
        for seed in seeds:
            args.SEED = seed
            np.random.seed(args.SEED)
            torch.manual_seed(args.SEED)
            torch.cuda.manual_seed_all(args.SEED)

            args.model_path  = './model/' + args.mode + str(args.MAX_LEN) + '_' + args.BASE + '_'+ str(args.lr) +'_' + '_batch_' \
                                + str(args.batch_size) + '_personality_' + personality + '_seed_' + str(seed) + 'f1/'

            train_dataloader, valid_dataloader, test_dataloader, train_length = load_data(df, args, tokenizer)

            
            if args.mode == 'Uttr':
                '''
                We use the pre-trained models to encode the utterance 
                from the speakers for personality prediction through the classification head.
                '''
                if args.BASE == 'BERT':
                    model     = BertForSequenceClassification.from_pretrained('bert-base-uncased', \
                               num_labels=args.num_class).cuda(args.device)
                elif args.BASE == 'RoBERTa':
                    model = RobertaForSequenceClassification.from_pretrained('roberta-base', \
                                num_labels=args.num_class).cuda(args.device)
        
               

            elif args.mode == 'Context':
                '''
                We input the whole dialog into the encoder for personality prediction. 
                We indicated the utterance from the analyzed speaker and the context 
                by segment embeddings in the pre-trained models: 1 for utterances and 0 for dialog context. 
                '''
                model     = BertForSequenceClassification.from_pretrained('bert-base-uncased', \
                            num_labels=args.num_class).cuda(args.device)

            elif args.mode == 'Context_Hierarchical' or args.mode == 'Context_Hierarchical_emoberta_uttr':
                '''
                We first use BERT to encode each utterance in the first layer (also incorporate with a VAD regression model), 
                and then in the second layer we model the context...
                '''
                if args.BASE == 'BERT':
                    
                    bert_mode = 'Uttr'
                    bert_lr = '2e-05'
                    bert_batch_size = '32'
                    
                    pre_trained_bert_path = './model/' + bert_mode + str(args.MAX_LEN) + '_' + args.BASE + '_'+ bert_lr +'_' + '_batch_' \
                                + bert_batch_size + '_personality_' + personality + '_seed_' + str(seed)  + 'f1/'
                    
                    model = DialogVAD.from_pretrained(pre_trained_bert_path, args=args).cuda(args.device)
                
                elif args.BASE == 'RoBERTa':
                    bert_mode = 'Uttr'
                    bert_lr = '2e-05'
                    bert_batch_size = '32'
                    
                    pre_trained_roberta_path = './model/' + bert_mode + str(args.MAX_LEN) + '_' + args.BASE + '_'+ bert_lr +'_' + '_batch_' \
                                + bert_batch_size + '_personality_' + personality + '_seed_' + str(seed)  + 'f1/'
                    
                    model = DialogVAD_roberta.from_pretrained(pre_trained_roberta_path, args=args).cuda(args.device)


            training_loss, best_eval_acc = train_model(model, args, train_dataloader, valid_dataloader, train_length)
            
            if args.mode == 'Context' or args.mode == 'baseline_3.1':
                model = BertForSequenceClassification.from_pretrained(args.model_path, \
                       num_labels=args.num_class).cuda(args.device)
            
            elif args.mode == 'Context_Hierarchical' or args.mode == 'Context_Hierarchical_emoberta_uttr':
                if args.BASE == 'BERT':
                    model     = DialogVAD.from_pretrained(args.model_path, args=args).cuda(args.device)
                elif args.BASE == 'RoBERTa':
                    model     = DialogVAD_roberta.from_pretrained(args.model_path, args=args).cuda(args.device)
            
            else:
                if args.BASE == 'BERT':
                   model = BertForSequenceClassification.from_pretrained(args.model_path, \
                       num_labels=args.num_class).cuda(args.device)
                elif args.BASE == 'RoBERTa':
                    model = RobertaForSequenceClassification.from_pretrained(args.model_path, \
                            num_labels=args.num_class).cuda(args.device)
                elif args.BASE == 'EmoBERTa':
                    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_class).cuda(args.device)
                
                

            print('Load model from', args.model_path)
            test_acc = eval_model(model, args, test_dataloader)
            test_acc_all_seeds.append(test_acc)
            print('Current Seed is', seed)
            print('Test acc:', test_acc)
            print('*'* 10, test_acc_total)
            print()
            
        test_acc_total.append(test_acc_all_seeds)
        print('\n========\n')
        print(test_acc_total)
    f.write(str(test_acc_total))




