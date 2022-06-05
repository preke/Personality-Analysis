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
from train import train_model, eval_model, train_model_again
from model import Baseline_1, Baseline_2, Uttr_VAD_embedding, DialogVAD, Baseline_1_roberta, DialogVAD_roberta
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
# CONFIG

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()

args.device       = 0

args.SEED          = 42
args.MAX_LEN       = 64
args.MAX_NUM_UTTR  = 30
args.batch_size    = 16
args.adam_epsilon  = 1e-8
args.epochs        = 3
args.num_class     = 2
args.drop_out      = 0.1
args.test_size     = 0.1
args.d_transformer = 64


args.mode         = 'Context_Hierarchical'
args.BASE         = 'BERT'
args.VAD_tokenized_dict = '../VAD_tokenized_dict.json'
args.result_name  = args.mode + '.txt' 





## get vad dict
VAD_Lexicons = pd.read_csv('../data/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt', sep='\t')
VAD_dict = {}
for r in VAD_Lexicons.iterrows():
    VAD_dict[r[1]['Word']] = [r[1]['Valence'], r[1]['Arousal'], r[1]['Dominance']]
args.VAD_dict = VAD_dict


personalities = ['A','C','E','O','N']


from transformers import AutoTokenizer, AutoModelForSequenceClassification

if args.BASE == 'BERT':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    epoch_list = [3]
    lr_list = [1e-5]
elif args.BASE == 'RoBERTa':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    epoch_list = [3]
    lr_list = [1e-6]
# elif args.BASE == 'EmoBERTa':
#     tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base", do_lower_case=True)
#     epoch_list = [10]
#     lr_list = [1e-4]

args.lr            = lr_list[0]
args.model_path   = './model/' + args.mode + str(args.MAX_LEN) + '_' + args.BASE + '_'+ str(args.lr ) +'_' + '_batch16/'



# personalities = ['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A',
#                  'C','C','C','C','C','C','C','C','C','C','C','C','C','C','C',
#                  'E','E','E','E','E','E','E','E','E','E','E','E','E','E','E',
#                  'O','O','O','O','O','O','O','O','O','O','O','O','O','O','O',
#                  'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N']
# lr_list = [1e-6,2e-6,3e-6,4e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,1e-4,2e-4,3e-4,4e-4,5e-4,
#     1e-6,2e-6,3e-6,4e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,1e-4,2e-4,3e-4,4e-4,5e-4,
#     1e-6,2e-6,3e-6,4e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,1e-4,2e-4,3e-4,4e-4,5e-4,
#     1e-6,2e-6,3e-6,4e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,1e-4,2e-4,3e-4,4e-4,5e-4,
#     1e-6,2e-6,3e-6,4e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,1e-4,2e-4,3e-4,4e-4,5e-4]
# epoch_list = [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
#         20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
#         20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
#         20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
#         20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]

cnt = 0

seeds = [0]# [0, 1, 13, 41, 42, 123, 456, 321, 999, 1024]

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
            train_dataloader, valid_dataloader, test_dataloader, train_length = load_data(df, args, tokenizer)

            
            if args.mode == 'Ours':
                model     = Baseline_2.from_pretrained('bert-base-uncased').cuda(args.device)
                
            elif args.mode == 'Uttr':
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
                elif args.BASE == 'EmoBERTa':
                    model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base", \
                        num_labels=args.num_class).cuda(args.device)
        
               
            elif args.mode == 'Uttr_VAD':
                '''
                Beside the Uttr for personality prediction, we add 
                a VAD regression task to supervised the model to 
                extract the affective information through a multi-task learning scheme.
                '''
                
                if args.BASE == 'BERT':
                    model     = Baseline_1.from_pretrained('bert-base-uncased').cuda(args.device)
                elif args.BASE == 'RoBERTa':
                    model     = Baseline_1_roberta.from_pretrained('roberta-base').cuda(args.device)

            elif args.mode == 'Context':
                '''
                We input the whole dialog into the encoder for personality prediction. 
                We indicated the utterance from the analyzed speaker and the context 
                by segment embeddings in the pre-trained models: 1 for utterances and 0 for dialog context. 
                '''
                model     = BertForSequenceClassification.from_pretrained('bert-base-uncased', \
                            num_labels=args.num_class).cuda(args.device)
                
            elif args.mode == 'Context_VAD':
                '''
                Based on Context, we also add the additional VAD regression task
                for the sum of all the words in a dialog flow in VAD dimensions.
                '''
                model     = Baseline_1.from_pretrained('bert-base-uncased').cuda(args.device)

            elif args.mode == 'Uttr_VAD_embedding':
                '''
                We use the VAD vector of each word in the utterance as word embedding input of the pre-trained model,
                instead of the look-up embedding. 
                可能存在的问题是，如果用tokenize的话，如果words被拆分为sub-words之后就没有VAD vector了...
                '''
                model     = Uttr_VAD_embedding.from_pretrained('bert-base-uncased').cuda(args.device)

                
            elif args.mode == 'Context_Hierarchical':
                '''
                We first use BERT to encode each utterance in the first layer (also incorporate with a VAD regression model), 
                and then in the second layer we model the context...
                '''
                if args.BASE == 'BERT':
                    model     = DialogVAD.from_pretrained('bert-base-uncased', args=args).cuda(args.device)
                elif args.BASE == 'RoBERTa':
                    model     = DialogVAD_roberta.from_pretrained('roberta-base').cuda(args.device)
                
            elif args.mode == 'Context_VAD_embedding':
                '''
                We input the whole dialog into the encoder for personality prediction. 
                We use the VAD vector of each word in the dialog as word embedding input of the pre-trained model,
                instead of the look-up embedding. 
                模型和Uttr_VAD_embedding是一样的，只是长度变长
                '''
                model     = Uttr_VAD_embedding.from_pretrained('bert-base-uncased').cuda(args.device)


            training_loss,best_eval_acc = train_model(model, args, train_dataloader, valid_dataloader, train_length)
            
            
            # ===== 
            
            # after training on multi-task, train again only on the target task
            # best_eval_acc = 0.6043956043956044
            
            
            # args.epochs = 10
            # args.lr = 1e-4
            # model     = DialogVAD.from_pretrained(args.model_path).cuda(args.device)
            # training_loss = train_model_again(model, args, train_dataloader, valid_dataloader, train_length, best_eval_acc)
            
            # =====             
            
            

            if args.mode == 'Ours':
                model     = Baseline_2.from_pretrained(args.model_path).cuda(args.device)
            
            elif args.mode == 'Uttr_VAD_embedding':
                model     = Uttr_VAD_embedding.from_pretrained(args.model_path).cuda(args.device)
            
            elif args.mode == 'Context_VAD_embedding':
                model     = Uttr_VAD_embedding.from_pretrained(args.model_path).cuda(args.device)
            
            elif args.mode == 'Context_VAD':
                model     = Baseline_1.from_pretrained(args.model_path).cuda(args.device)
            
            elif args.mode == 'Uttr_VAD':
                if args.BASE == 'BERT':
                    model     = Baseline_1.from_pretrained(args.model_path).cuda(args.device)
                elif args.BASE == 'RoBERTa':
                    model     = Baseline_1_roberta.from_pretrained(args.model_path).cuda(args.device)
            
            elif args.mode == 'Context' or args.mode == 'baseline_3.1':
                model = BertForSequenceClassification.from_pretrained(args.model_path, \
                       num_labels=args.num_class).cuda(args.device)
            
            elif args.mode == 'Context_Hierarchical':
                if args.BASE == 'BERT':
                    model     = DialogVAD.from_pretrained(args.model_path, args=args).cuda(args.device)
                elif args.BASE == 'RoBERTa':
                    model     = DialogVAD_roberta.from_pretrained(args.model_path).cuda(args.device)
            
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
            
        test_acc_total.append(test_acc_all_seeds)
        print('\n========\n')
        print(test_acc_total)
    f.write(str(test_acc_total))




