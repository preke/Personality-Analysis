import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,IterableDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import json
import re





# class Preprocessing:

def get_vad(VAD_dict, sents):
    VAD_scores = []
    for sent in sents:
        w_list = sent.split()
        v_score, a_score, d_score = [], [], []
        for word in w_list:
            try:
                v_score.append(VAD_dict[word][0])
                a_score.append(VAD_dict[word][1])
                d_score.append(VAD_dict[word][2])
            except:
                v_score.append(0)
                a_score.append(0)
                d_score.append(0)
        VAD_scores.append([v_score, a_score, d_score])
    return VAD_scores

def get_VAD_tokenized_dict(i, VAD_tokenized_dict):
    try:
        return VAD_tokenized_dict[i]
    except:
        return [0.0,0.0,0.0]


def get_seg_id(sent_list, role):
    '''
    Generate the segment id for the whole sent
    '''
    
    ans = []
    for i in eval(sent_list):
        if i[0].split(' ')[0] != role:
            
            tmp = [0]*(len(i[1].split())+1)
            ans += tmp
        else:
            tmp = [1]*(len(i[1].split())+1)
            ans += tmp
    return ans

def get_sent(sent_list, role):
    '''
    Obtain the whole sent
    '''
    
    ans = ""
    for i in eval(sent_list):
        if i[0].split(' ')[0] != role:
            ans = ans + " " + i[1]
        else:
            ans = ans + " " + i[1]
    return ans


def padding_uttrs(contexts, padding_element, args):
    ans_contexts = []
    
    for sents in contexts:
        pad_num = args.MAX_NUM_UTTR - len(sents) 
        if pad_num > 0: # e.g. 30 - 15 
            for i in range(pad_num):
                sents.append(padding_element)
        elif pad_num < 0: # e.g. 30 - 36 
            sents = sents[:args.MAX_NUM_UTTR]
        ans_contexts.append(sents)

    return ans_contexts


def load_data(df, args, tokenizer):
    if args.mode == 'Context':
        '''
        We input the whole dialog into the encoder for personality prediction. 
        We indicated the utterance from the analyzed speaker and the context 
        by segment embeddings in the pre-trained models: 1 for utterances and 0 for dialog context. 
        '''

        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN/4 + 1), \
            pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        contexts = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['context']]
        context_masks = [[float(i>0) for i in seq] for seq in contexts]

        sents = []
        sent_masks = []
        sent_seg_embeddings = []

        for i in range(len(uttrs)):
            sents.append(uttrs[i] + contexts[i][1:]) ## remove the latter [CLS]
            sent_masks.append(uttr_masks[i] + context_masks[i][1:])
            sent_seg_embeddings.append([1]*len(uttrs[i]) + [0]*len(contexts[i][1:]))
        
        labels = list(df['labels'])

        train_sents, test_sents, train_labels, test_labels = \
            train_test_split(sents, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_sent_masks, test_sent_masks,_,_ = train_test_split(sent_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_seg_embeddings, test_seg_embeddings,_,_ = train_test_split(sent_seg_embeddings,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)


        train_set_labels = train_labels


        train_sents, valid_sents, train_labels, valid_labels = \
            train_test_split(train_sents, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_sent_masks, valid_sent_masks,_,_ = train_test_split(train_sent_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_seg_embeddings, valid_seg_embeddings,_,_ = train_test_split(train_seg_embeddings, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_sents          = torch.tensor(train_sents)
        valid_sents          = torch.tensor(valid_sents)
        test_sents           = torch.tensor(test_sents)

        train_sent_masks     = torch.tensor(train_sent_masks)
        valid_sent_masks     = torch.tensor(valid_sent_masks)
        test_sent_masks      = torch.tensor(test_sent_masks)

        train_seg_embeddings = torch.tensor(train_seg_embeddings)
        valid_seg_embeddings = torch.tensor(valid_seg_embeddings)
        test_seg_embeddings  = torch.tensor(test_seg_embeddings)

        train_labels         = torch.tensor(train_labels)    
        valid_labels         = torch.tensor(valid_labels)
        test_labels          = torch.tensor(test_labels)

        train_data       = TensorDataset(train_sents, train_sent_masks, train_seg_embeddings, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_sents, valid_sent_masks, valid_seg_embeddings, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_sents, test_sent_masks, test_seg_embeddings, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length



        # ---------
    elif args.mode == 'Context_VAD':
        '''
        Based on Context, we also add the additional VAD regression task
        for the sum of all the words in a dialog flow in VAD dimensions.
        '''

        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN/4 + 1), \
            pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        contexts = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['context']]
        context_masks = [[float(i>0) for i in seq] for seq in contexts]

        sents = []
        sent_masks = []
        sent_seg_embeddings = []




        for i in range(len(uttrs)):
            sents.append(uttrs[i] + contexts[i][1:]) ## remove the latter [CLS]
            sent_masks.append(uttr_masks[i] + context_masks[i][1:])
            sent_seg_embeddings.append([1]*len(uttrs[i]) + [0]*len(contexts[i][1:]))
        
        vad_scores = get_vad(args.VAD_dict, sents, tokenizer)

        
        labels = list(df['labels'])

        train_sents, test_sents, train_labels, test_labels = \
            train_test_split(sents, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_sent_masks, test_sent_masks,_,_ = train_test_split(sent_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_seg_embeddings, test_seg_embeddings,_,_ = train_test_split(sent_seg_embeddings,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)


        train_vads, test_vads,_,_ = train_test_split(vad_scores, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        

        train_set_labels = train_labels


        train_sents, valid_sents, train_labels, valid_labels = \
            train_test_split(train_sents, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_sent_masks, valid_sent_masks,_,_ = train_test_split(train_sent_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_seg_embeddings, valid_seg_embeddings,_,_ = train_test_split(train_seg_embeddings, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)

        train_vads, valid_vads,_,_ = train_test_split(train_vads, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_sents          = torch.tensor(train_sents).cuda(args.device)
        valid_sents          = torch.tensor(valid_sents).cuda(args.device)
        test_sents           = torch.tensor(test_sents).cuda(args.device)

        train_sent_masks     = torch.tensor(train_sent_masks).cuda(args.device)
        valid_sent_masks     = torch.tensor(valid_sent_masks).cuda(args.device)
        test_sent_masks      = torch.tensor(test_sent_masks).cuda(args.device)

        train_seg_embeddings = torch.tensor(train_seg_embeddings).cuda(args.device)
        valid_seg_embeddings = torch.tensor(valid_seg_embeddings).cuda(args.device)
        test_seg_embeddings  = torch.tensor(test_seg_embeddings).cuda(args.device)
        

        train_labels         = torch.tensor(train_labels).cuda(args.device)    
        valid_labels         = torch.tensor(valid_labels).cuda(args.device)
        test_labels          = torch.tensor(test_labels).cuda(args.device)

        train_vads           = torch.tensor(train_vads).cuda(args.device)    
        valid_vads           = torch.tensor(valid_vads).cuda(args.device)
        test_vads            = torch.tensor(test_vads).cuda(args.device)

        train_data       = TensorDataset(train_sents, train_sent_masks, train_seg_embeddings, train_vads, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_sents, valid_sent_masks, valid_seg_embeddings, valid_vads, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_sents, test_sent_masks, test_seg_embeddings, test_vads, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length

    elif args.mode == 'Uttr_VAD':

        '''
        Based on Context, we also add the additional VAD regression task
        for the sum of all the words in a dialog flow in VAD dimensions.
        '''

        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN/4 + 1), \
            pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]


        
        vad_scores = get_vad(args.VAD_dict, uttrs, tokenizer)
        
        labels = list(df['labels'])

        train_uttrs, test_uttrs, train_labels, test_labels = \
            train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        

        train_vads, test_vads,_,_ = train_test_split(vad_scores, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        

        train_set_labels = train_labels


        train_uttrs, valid_uttrs, train_labels, valid_labels = \
            train_test_split(train_uttrs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        train_vads, valid_vads,_,_ = train_test_split(train_vads, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_uttrs          = torch.tensor(train_uttrs).cuda(args.device)
        valid_uttrs          = torch.tensor(valid_uttrs).cuda(args.device)
        test_uttrs           = torch.tensor(test_uttrs).cuda(args.device)

        train_uttr_masks     = torch.tensor(train_uttr_masks).cuda(args.device)
        valid_uttr_masks     = torch.tensor(valid_uttr_masks).cuda(args.device)
        test_uttr_masks      = torch.tensor(test_uttr_masks).cuda(args.device)

        train_labels         = torch.tensor(train_labels).cuda(args.device)    
        valid_labels         = torch.tensor(valid_labels).cuda(args.device)
        test_labels          = torch.tensor(test_labels).cuda(args.device)

        train_vads           = torch.tensor(train_vads).cuda(args.device)    
        valid_vads           = torch.tensor(valid_vads).cuda(args.device)
        test_vads            = torch.tensor(test_vads).cuda(args.device)

        train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_vads, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_uttrs, valid_uttr_masks, valid_vads, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_vads, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length

    elif args.mode == 'Uttr_VAD_embedding':
        '''
        We use the VAD vector of each word in the utterance as word embedding input of the pre-trained model.
        So, we need to load the VAD vector for each word.
        '''
        
        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['utterance']]
        
        with open(args.VAD_tokenized_dict) as json_file:
            VAD_tokenized_dict = json.load(json_file)

        uttr_vads = [[get_VAD_tokenized_dict(i, VAD_tokenized_dict) for i in j] for j in uttrs]    
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]

        labels = list(df['labels'])

        train_uttr_vads, test_uttr_vads, train_labels, test_labels = \
            train_test_split(uttr_vads, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)

        train_set_labels = train_labels

        train_uttr_vads, valid_uttr_vads, train_labels, valid_labels = \
            train_test_split(train_uttr_vads, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks,train_set_labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)

        train_uttr_vads  = torch.tensor(train_uttr_vads)
        valid_uttr_vads  = torch.tensor(valid_uttr_vads)
        test_uttr_vads   = torch.tensor(test_uttr_vads)

        train_uttr_masks = torch.tensor(train_uttr_masks)
        valid_uttr_masks = torch.tensor(valid_uttr_masks)
        test_uttr_masks  = torch.tensor(test_uttr_masks)
        
        train_labels         = torch.tensor(train_labels).cuda(args.device)    
        valid_labels         = torch.tensor(valid_labels).cuda(args.device)
        test_labels          = torch.tensor(test_labels).cuda(args.device)
        
        train_data       = TensorDataset(train_uttr_vads, train_uttr_masks, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_uttr_vads, valid_uttr_masks, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_uttr_vads, test_uttr_masks, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length

    elif args.mode == 'Context_VAD_embedding':
        '''
        We input the whole dialog into the encoder for personality prediction. 
        We use the VAD vector of each word in the dialog as word embedding input of the pre-trained model,
        instead of the look-up embedding. 
        模型和Uttr_VAD_embedding是一样的，只是长度变长
        '''

        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN/4 + 1), \
            pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        contexts = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['context']]
        context_masks = [[float(i>0) for i in seq] for seq in contexts]


        sents = []
        sent_masks = []
        sent_seg_embeddings = []


        for i in range(len(uttrs)):
            sents.append(uttrs[i] + contexts[i][1:]) ## remove the latter [CLS]
            sent_masks.append(uttr_masks[i] + context_masks[i][1:])
            sent_seg_embeddings.append([1]*len(uttrs[i]) + [0]*len(contexts[i][1:]))

        with open(args.VAD_tokenized_dict) as json_file:
            VAD_tokenized_dict = json.load(json_file)

        sent_vads = [[get_VAD_tokenized_dict(i, VAD_tokenized_dict) for i in j] for j in sents]    


        
        labels = list(df['labels'])

        train_sent_vads, test_sent_vads, train_labels, test_labels = \
            train_test_split(sent_vads, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_sent_masks, test_sent_masks,_,_ = train_test_split(sent_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_seg_embeddings, test_seg_embeddings,_,_ = train_test_split(sent_seg_embeddings,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)


        train_set_labels = train_labels


        train_sent_vads, valid_sent_vads, train_labels, valid_labels = \
            train_test_split(train_sent_vads, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_sent_masks, valid_sent_masks,_,_ = train_test_split(train_sent_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_seg_embeddings, valid_seg_embeddings,_,_ = train_test_split(train_seg_embeddings, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_sent_vads          = torch.tensor(train_sent_vads)
        valid_sent_vads          = torch.tensor(valid_sent_vads)
        test_sent_vads           = torch.tensor(test_sent_vads)

        train_sent_masks     = torch.tensor(train_sent_masks)
        valid_sent_masks     = torch.tensor(valid_sent_masks)
        test_sent_masks      = torch.tensor(test_sent_masks)

        train_seg_embeddings = torch.tensor(train_seg_embeddings)
        valid_seg_embeddings = torch.tensor(valid_seg_embeddings)
        test_seg_embeddings  = torch.tensor(test_seg_embeddings)

        train_labels         = torch.tensor(train_labels)    
        valid_labels         = torch.tensor(valid_labels)
        test_labels          = torch.tensor(test_labels)

        train_data       = TensorDataset(train_sent_vads, train_sent_masks, train_seg_embeddings, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_sent_vads, valid_sent_masks, valid_seg_embeddings, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_sent_vads, test_sent_masks, test_seg_embeddings, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length


        ## ----
        
    elif args.mode == 'Ours':

        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN), \
            pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        print(tokenizer.decode(uttrs[0]))
        
        contexts = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN), \
                pad_to_max_length=True) for sent in df['context']]
        context_masks = [[float(i>1) for i in seq] for seq in contexts]

        sents = []
        sent_masks = []
        sent_seg_embeddings = []

        for i in range(len(uttrs)):
            half_len = int(len(uttrs[i])/2)
            sents.append(uttrs[i][:half_len] + contexts[i][:half_len])
            sent_masks.append(uttr_masks[i][:half_len] + context_masks[i][:half_len])
            sent_seg_embeddings.append([1]*len(uttrs[i][:half_len]) + [0]*len(contexts[i][:half_len]))
        
        print(tokenizer.decode(sents[0]))
        
        vad_scores = get_vad(args.VAD_dict, uttrs, tokenizer)
        
        
        
        
        labels = list(df['labels'])

        train_uttrs, test_uttrs, train_labels, test_labels = \
            train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
        train_sents, test_sents, train_labels, test_labels = \
            train_test_split(sents, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_sent_masks, test_sent_masks,_,_ = train_test_split(sent_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_seg_embeddings, test_seg_embeddings,_,_ = train_test_split(sent_seg_embeddings,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)


        train_vads, test_vads,_,_ = train_test_split(vad_scores, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        

        train_set_labels = train_labels


        train_uttrs, valid_uttrs, train_labels, valid_labels = \
            train_test_split(train_uttrs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        

        train_sents, valid_sents, train_labels, valid_labels = \
            train_test_split(train_sents, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_sent_masks, valid_sent_masks,_,_ = train_test_split(train_sent_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_seg_embeddings, valid_seg_embeddings,_,_ = train_test_split(train_seg_embeddings, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)

        train_vads, valid_vads,_,_ = train_test_split(train_vads, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_uttrs          = torch.LongTensor(train_uttrs)
        valid_uttrs          = torch.LongTensor(valid_uttrs)
        test_uttrs           = torch.LongTensor(test_uttrs)

        train_uttr_masks     = torch.LongTensor(train_uttr_masks)
        valid_uttr_masks     = torch.LongTensor(valid_uttr_masks)
        test_uttr_masks      = torch.LongTensor(test_uttr_masks)

        train_sents          = torch.LongTensor(train_sents)
        valid_sents          = torch.LongTensor(valid_sents)
        test_sents           = torch.LongTensor(test_sents)

        train_sent_masks     = torch.LongTensor(train_sent_masks)
        valid_sent_masks     = torch.LongTensor(valid_sent_masks)
        test_sent_masks      = torch.LongTensor(test_sent_masks)

        train_seg_embeddings = torch.LongTensor(train_seg_embeddings)
        valid_seg_embeddings = torch.LongTensor(valid_seg_embeddings)
        test_seg_embeddings  = torch.LongTensor(test_seg_embeddings)
        

        train_labels         = torch.tensor(train_labels)    
        valid_labels         = torch.tensor(valid_labels)
        test_labels          = torch.tensor(test_labels)

        train_vads           = torch.tensor(train_vads)    
        valid_vads           = torch.tensor(valid_vads)
        test_vads            = torch.tensor(test_vads)

        train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_sents, train_sent_masks, train_seg_embeddings, train_vads, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_uttrs, valid_uttr_masks, valid_sents, valid_sent_masks, valid_seg_embeddings, valid_vads, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_sents, test_sent_masks, test_seg_embeddings, test_vads, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length

    elif args.mode == 'baseline_3.1':


        df['sent']   = df.apply(lambda r: get_sent(r['text'], r['character']), axis=1)
        df['seg_id'] = df.apply(lambda r: get_seg_id(r['text'], r['character']), axis=1)

        sents = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
            pad_to_max_length=True) for sent in df['sent']]
        sent_masks = [[float(i>0) for i in seq] for seq in sents]


        sent_seg_embeddings = [i for i in df['seg_id']]
        
        labels              = list(df['labels'])

        
        train_sents, test_sents, train_labels, test_labels = \
            train_test_split(sents, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_sent_masks, test_sent_masks,_,_ = train_test_split(sent_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_seg_embeddings, test_seg_embeddings,_,_ = train_test_split(sent_seg_embeddings,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)


        train_set_labels = train_labels


        train_sents, valid_sents, train_labels, valid_labels = \
            train_test_split(train_sents, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_sent_masks, valid_sent_masks,_,_ = train_test_split(train_sent_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_seg_embeddings, valid_seg_embeddings,_,_ = train_test_split(train_seg_embeddings, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_sents          = torch.tensor(train_sents)
        valid_sents          = torch.tensor(valid_sents)
        test_sents           = torch.tensor(test_sents)

        train_sent_masks     = torch.tensor(train_sent_masks)
        valid_sent_masks     = torch.tensor(valid_sent_masks)
        test_sent_masks      = torch.tensor(test_sent_masks)
        
        print(len(train_sents))
        print(train_sents.size())
        print(train_seg_embeddings[0])
        
        
        train_seg_embeddings = torch.tensor(train_seg_embeddings)
        valid_seg_embeddings = torch.tensor(valid_seg_embeddings)
        test_seg_embeddings  = torch.tensor(test_seg_embeddings)

        train_labels         = torch.tensor(train_labels)    
        valid_labels         = torch.tensor(valid_labels)
        test_labels          = torch.tensor(test_labels)

        train_data       = TensorDataset(train_sents, train_sent_masks, train_seg_embeddings, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_sents, valid_sent_masks, valid_seg_embeddings, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_sents, test_sent_masks, test_seg_embeddings, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length

    elif args.mode == 'Context_Hierarchical':
        
        ## args.MAX_NUM_UTTR 

        dialog_context = df['raw_text'].apply(lambda x: [i[1] for i in eval(x)])

        contexts       = [[tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True) for sent in sents] for sents in dialog_context]
        context_masks  = [[[float(i>0) for i in seq] for seq in sents] for sents in contexts]

        dialog_states  = [eval(i) for i in df['dialog_state']]
        labels         = list(df['labels'])
        uttr_vads      = [get_vad(args.VAD_dict, sent) for sent in dialog_context]
        
        print(len(contexts))
        print([len(i) for i in contexts])
        print(uttr_vads[0])
        

        #vad_scores = torch.Tensor(vad_scores)
        #print(vad_scores.shape)
        import time
        time.sleep(100)

        contexts      = padding_uttrs(contexts, [0]*args.MAX_LEN, args) # padding_element: [PAD]
        context_masks = padding_uttrs(context_masks, [0]*args.MAX_LEN, args) # padding_element: [PAD]
        
        dialog_states = padding_uttrs(dialog_states, -1, args)
        uttr_vads     = padding_uttrs(uttr_vads, [0.0, 0.0, 0.0], args)

        
        print('-------------------------------------')



        train_contexts, test_contexts, train_labels, test_labels = \
            train_test_split(contexts, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        
        train_context_masks, test_context_masks,_,_ = train_test_split(context_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)

        train_uttr_vads, test_uttr_vads,_,_ = train_test_split(uttr_vads, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)

        train_dialog_states, test_dialog_states,_,_ = train_test_split(dialog_states, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)

        train_set_labels = train_labels


        train_contexts, valid_contexts, train_labels, valid_labels = \
            train_test_split(train_contexts, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        
        train_context_masks, valid_context_masks,_,_ = train_test_split(train_context_masks, train_set_labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)

        train_uttr_vads, valid_uttr_vads,_,_ = train_test_split(train_uttr_vads, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)

        train_dialog_states, valid_dialog_states,_,_ = train_test_split(train_dialog_states, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        
        

        train_contexts      = torch.tensor(train_contexts) # [torch.tensor(i) for i in train_contexts]
        valid_contexts      = torch.tensor(valid_contexts) # [torch.tensor(i) for i in valid_contexts]
        test_contexts       = torch.tensor(test_contexts)  # [torch.tensor(i) for i in test_contexts]

        train_context_masks = torch.tensor(train_context_masks) # [torch.tensor(i) for i in train_context_masks]
        valid_context_masks = torch.tensor(valid_context_masks) # [torch.tensor(i) for i in valid_context_masks]
        test_context_masks  = torch.tensor(test_context_masks)  #[torch.tensor(i) for i in test_context_masks]
        
        train_uttr_vads      = torch.tensor(train_uttr_vads) # [torch.tensor(i) for i in train_uttr_vads]
        valid_uttr_vads      = torch.tensor(valid_uttr_vads) # [torch.tensor(i) for i in valid_uttr_vads]
        test_uttr_vads       = torch.tensor(test_uttr_vads)  # [torch.tensor(i) for i in test_uttr_vads]

        train_dialog_states  = torch.tensor(train_dialog_states) # [torch.tensor(i) for i in train_dialog_states]
        valid_dialog_states  = torch.tensor(valid_dialog_states) # [torch.tensor(i) for i in valid_dialog_states]
        test_dialog_states   = torch.tensor(test_dialog_states)  # [torch.tensor(i) for i in test_dialog_states]

        train_labels        = torch.tensor(train_labels)    
        valid_labels        = torch.tensor(valid_labels)
        test_labels         = torch.tensor(test_labels)

        train_data       = TensorDataset(train_contexts, train_context_masks, train_uttr_vads, train_dialog_states, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_contexts, valid_context_masks, valid_uttr_vads, valid_dialog_states, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_contexts, test_context_masks, test_uttr_vads, test_dialog_states, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length
        # return train_dataloader, test_dataloader, train_length
    
    
    elif args.mode == 'Uttr':
        
        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        labels = list(df['labels'])
        
        train_uttrs, test_uttrs, train_labels, test_labels = \
            train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
        train_set_labels = train_labels
        
        train_uttrs, valid_uttrs, train_labels, valid_labels = \
            train_test_split(train_uttrs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        train_uttrs         = torch.tensor(train_uttrs)
        valid_uttrs         = torch.tensor(valid_uttrs)
        test_uttrs          = torch.tensor(test_uttrs)

        train_uttr_masks    = torch.tensor(train_uttr_masks)
        valid_uttr_masks    = torch.tensor(valid_uttr_masks)
        test_uttr_masks     = torch.tensor(test_uttr_masks)
        
        train_labels        = torch.tensor(train_labels)    
        valid_labels        = torch.tensor(valid_labels)
        test_labels         = torch.tensor(test_labels)


        train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_uttrs, valid_uttr_masks, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length
        '''
        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['utterance']]

        uttr_masks = [[float(i>1) for i in seq] for seq in uttrs]
        
#         print(tokenizer.decode(0))
#         print(tokenizer.decode(1437))
        
#         print(df['utterance'][0])
#         print(uttrs[0])
#         print(tokenizer.decode(uttrs[0]))
#         print(uttr_masks[0])
        
#         print(df['utterance'][1])
#         print(uttrs[1])
#         print(tokenizer.decode(uttrs[1]))        
#         print(uttr_masks[1])    
        
#         print(df['utterance'][2])
#         print(uttrs[2])
#         print(tokenizer.decode(uttrs[2]))        
#         print(uttr_masks[2])  
        
        
        labels = list(df['labels'])
        
        train_uttrs, test_uttrs, train_labels, test_labels = \
            train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
       
        
        
        train_uttrs         = torch.tensor(train_uttrs)
        test_uttrs          = torch.tensor(test_uttrs)

        train_uttr_masks    = torch.tensor(train_uttr_masks)
        test_uttr_masks     = torch.tensor(test_uttr_masks)
        
        train_labels        = torch.tensor(train_labels)    
        test_labels         = torch.tensor(test_labels)


        train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        
        test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, test_dataloader, train_length
        '''
        
    else:

        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        contexts = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['context']]
        context_masks = [[float(i>0) for i in seq] for seq in contexts]

        
        uttr_vad_scores = get_vad(args.VAD_dict, uttrs, tokenizer)
        context_vad_scores = get_vad(args.VAD_dict, contexts, tokenizer)
        
        
        labels = list(df['labels'])
        
        train_uttrs, test_uttrs, train_labels, test_labels = \
            train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
        train_contexts, test_contexts,_,_ = train_test_split(contexts, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_context_masks, test_context_masks,_,_ = train_test_split(context_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
        train_uttr_vad, test_uttr_vad,_,_ = train_test_split(uttr_vad_scores, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_context_vad, test_context_vad,_,_ = train_test_split(context_vad_scores, labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)

        train_set_labels = train_labels

        train_uttrs, valid_uttrs, train_labels, valid_labels = \
            train_test_split(train_uttrs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        train_contexts, valid_contexts,_,_ = train_test_split(train_contexts, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_context_masks, valid_context_masks,_,_ = train_test_split(train_context_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        train_uttr_vad, valid_uttr_vad,_,_ = train_test_split(train_uttr_vad, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_context_vad, valid_context_vad,_,_ = train_test_split(train_context_vad, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        train_uttrs         = torch.tensor(train_uttrs)
        valid_uttrs         = torch.tensor(valid_uttrs)
        test_uttrs          = torch.tensor(test_uttrs)

        train_uttr_masks    = torch.tensor(train_uttr_masks)
        valid_uttr_masks    = torch.tensor(valid_uttr_masks)
        test_uttr_masks     = torch.tensor(test_uttr_masks)

        train_contexts      = torch.tensor(train_contexts)
        valid_contexts      = torch.tensor(valid_contexts)
        test_contexts       = torch.tensor(test_contexts)

        train_context_masks = torch.tensor(train_context_masks)
        valid_context_masks = torch.tensor(valid_context_masks)
        test_context_masks  = torch.tensor(test_context_masks)
        
        train_uttr_vad      = torch.tensor(train_uttr_vad)
        valid_uttr_vad      = torch.tensor(valid_uttr_vad)
        test_uttr_vad       = torch.tensor(test_uttr_vad)

        train_context_vad   = torch.tensor(train_context_vad)
        valid_context_vad   = torch.tensor(valid_context_vad)
        test_context_vad    = torch.tensor(test_context_vad)

        train_labels        = torch.tensor(train_labels)    
        valid_labels        = torch.tensor(valid_labels)
        test_labels         = torch.tensor(test_labels)

        
        train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_contexts, train_context_masks, \
                                            train_uttr_vad, train_context_vad, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_uttrs, valid_uttr_masks, valid_contexts, valid_context_masks, \
                                            valid_uttr_vad, valid_context_vad, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_contexts, test_context_masks, \
                                            test_uttr_vad, test_context_vad, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length