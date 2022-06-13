# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback
from data_loader import load_data
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,IterableDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# CONFIG

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()
args.device        = 0
args.MAX_LEN       = 128
args.batch_size    = 32
args.adam_epsilon  = 1e-8
args.epochs        = 3
args.num_class     = 7
args.test_size     = 0.1



personalities = ['A']#,'C','E','O','N']


tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
df = pd.read_csv('../data/Friends_'+personalities[0]+'_whole.tsv', sep='\t')

uttrs      = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True) for sent in df['utterance']]
uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
uttrs      = torch.tensor(uttrs)
uttr_masks      = torch.tensor(uttr_masks)

data = TensorDataset(uttrs, uttr_masks)
sampler    = RandomSampler(data)
dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base")


pred_list = np.array([])
model.eval()
for batch in dataloader:
    batch     = tuple(t.cuda(args.device) for t in batch)
    outputs   = model(uttrs, attention_mask=uttr_masks)
    logits    = outputs.logits
    pred_flat = np.argmax(logits, axis=1).flatten()
    pred_list = np.append(pred_list, pred_flat)


print(pred_list)






