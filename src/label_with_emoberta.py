# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback
from data_loader import load_data
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


tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-large")
df = pd.read_csv('../data/Friends_'+personality+'_whole.tsv', sep='\t')
uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True) for sent in df['utterance']]
uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]

model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-large")

def labeling(model, args, dataloader):
    model.eval()
    for batch in dataloader:
        batch = tuple(t.cuda(args.device) for t in batch)

        b_contexts, b_context_masks, b_vad_scores, b_dialog_states, b_labels = batch
        logits = model(b_contexts, b_context_masks, b_dialog_states, b_vad_scores)

        loss_ce             = nn.CrossEntropyLoss()
        classification_loss = loss_ce(logits, b_labels)
        loss                = classification_loss



