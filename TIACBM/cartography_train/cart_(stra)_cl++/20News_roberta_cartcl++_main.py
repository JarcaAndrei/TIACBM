from bs4 import BeautifulSoup
import pandas as pd
import os
import spacy_transformers
import json
import sys
import os
import torch
import pickle
import json
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import trange
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import bisect
import math
source_dir = './'
prefix = 'reuters' 
suffix = 'rand123'
model_name = 'news_with_pos4'
loss_func_name = 'base'

#if model_name == 'bert_base':
#    model_checkpoint = os.path.join(source_dir, 'berts', 'bert-base-uncased')

#data_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))
#data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
#labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
#class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
#train_num=pickle.load(open(os.path.join(source_dir, 'data', 'train_num.'+suffix),'rb'))
#num_labels = len(labels_ref)
from sklearn.datasets import fetch_20newsgroups
import json
file_path = r"sorted_20news_roberta.json"
sentences, labels = [], []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f: 
        data = json.loads(line.strip())  
        sentences.append(data["text"])
        labels.append(data["labels"])
data_val = fetch_20newsgroups(subset="test")
train_label = labels
val_label = data_val['target']
data_train = sentences
data_val = data_val['data']
class_freq = None
train_num = None
num_labels = 20
max_len = 512
lr = 5e-5 
epochs = 30
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", max_len=max_len)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=num_labels)).to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight'] 
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=lr) 
from util_loss import ResampleLoss

if loss_func_name == 'BCE':
    loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=class_freq, train_num=train_num)

if loss_func_name == 'FL':
    loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=class_freq, train_num=train_num)
    
if loss_func_name == 'CBloss': #CB
    loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                             class_freq=class_freq, train_num=train_num) 
    
if loss_func_name == 'R-BCE-Focal': # R-FL
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0, 
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                             class_freq=class_freq, train_num=train_num)

if loss_func_name == 'NTR-Focal': # NTR-FL
    loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             class_freq=class_freq, train_num=train_num)
    
if loss_func_name == 'DBloss-noFocal': # DB-0FL
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                             class_freq=class_freq, train_num=train_num)

if loss_func_name == 'CBloss-ntr': # CB-NTR
    loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                             class_freq=class_freq, train_num=train_num)
    
if loss_func_name == 'DBloss': # DB
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                             class_freq=class_freq, train_num=train_num)

loss_func = nn.CrossEntropyLoss()
        
from torch.utils.data import Dataset, DataLoader

def preprocess_function(docu):
    labels = docu['labels'] 
    encodings = tokenizer(docu['text'], truncation=True, padding='max_length')  
    return (torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']), torch.tensor(labels))

def get_synsets_for_word(word):
    synsets = wn.synsets(word)
    return synsets
def get_word_token_indices(tokens, word, tokenizer):
    word_tokens = tokenizer.tokenize(word)
    word_len = len(word_tokens)
    indices = []
    for i in range(len(tokens) - word_len + 1):
        if tokens[i:i + word_len] == word_tokens:
            indices.append(list(range(i, i + word_len)))
    return indices


class CustomDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return preprocess_function(self.documents[index])
from tqdm import tqdm
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

import random
import time
import spacy

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from num2words import num2words
nlp = spacy.load("en_core_web_lg") 
import datetime

def decade_to_words(decade_str):
    decade = int(decade_str[:4]) 
    decade_short = int(decade_str[2:3])  
    
    full_text = f"the {num2words(decade)}s" 
    
    return full_text

REMOVE_POS = {"ADP", "PUNCT", "DET",  "PRON", "CCONJ", "SCONJ", "PART", "X"}
def preprocess_num_token(token):
    if token == "'":
        return ''
    if token[0:2] == '-a':
        return token[1:]
    if ',s' in token:
        return num2words(float(str(token).replace(",s", "").replace(',','')))
    if token.strip('.') == '': 
        return ''
    if token[-1] == 's':
        if token.isalpha():
            return token
        try:
            return decade_to_words(token)
        except:
            return ''
    if token[-1] == '-':
        return str(token).replace("-", "")
    if '-' in token:
        if token[0].isalpha():
            return str(token).replace("-", " ")
        try:
            return date_to_words(token)
        except:
            return num2words(float(str(token).replace("-", "")))
    if ':' in token:
        if 'note:' in token:
            return 'note ' + num2words(float(str(token).replace('note:', '')))
        return time_to_words(token)
    if token[-1] == 'a' and '.' not in token:
        return num2words(float(str(token).replace("a", ""))) + " amendemnt"
    if token[-1] == 'p':
        if '/' in token:
            return str(token).replace('p', '') + ' pounds'
        return num2words(float(str(token).replace("p", ""))) + " pounds"
    if token.isalpha():
        return token
    if '/' in token:
        return token
    if ',' in token:
        token = token.replace(",","")
        if token[-1] == 'm':
            return num2words(float(str(token).replace("m", ""))) + ' millions'
        token=float(token)
    if str(token).replace(".","").isalpha():
        return token
    if "%" in str(token):
        return num2words(float(str(token).replace("%", ""))) + " percent"
    
    if "$" in str(token):
        return num2words(float(str(token).replace("$", ""))) + " dollars"

    if str(token)[-1] == 'k':
        return num2words(float(str(token).replace("k", ""))) + " thousand"
    try:
        return num2words(token)
    except:
        return ''
import re

#processed_data = [
#    {"text": text.lower(), "labels": label}
#    for text, label in zip(data_train, train_label)
#]

processed_val = [
    {"text": text.lower(), "labels": label}
    for text, label in zip(data_val, val_label)
]
#data_train = processed_data
data_val = processed_val

import pickle

class CustomDataset_train(Dataset):
    def __init__(self, documents, labels, stage, model, sentiment_dict=None):
        self.documents = documents
        self.labels = labels
        self.stage = stage
        self.device = 'cuda'
        self.sentiment_dict = sentiment_dict
        self.model = model
        self.encodings = []
        #self.docs = docs
        '''
        # Combine text and labels
        dataset = [{'text': text, 'label': label} for text, label in zip(self.documents, self.labels)]
        # Stratified sampling based on stage
        num_labels = len(set(labels))  # Total number of classes
        label_dict = {i: [] for i in range(num_labels)}
        
        for item in dataset:
            label_dict[item['label']].append(item)
        
        fractions = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1}
        
        stratified_data = []
        for label in label_dict:
            sampled_data = label_dict[label][:int(len(label_dict[label]) * fractions[self.stage])]
            stratified_data.extend(sampled_data)
        stratified_data = self._oversample(stratified_data)
        self.documents = [item['text'] for item in stratified_data]
        self.labels = [item['label'] for item in stratified_data]
        '''
        if stage == 1:
            self.documents = self.documents[:int(len(self.documents) * 0.6)]
            self.labels = self.labels[:int(len(self.labels) * 0.6)]
        elif stage == 2:
            self.documents = self.documents[:int(len(self.documents) * 0.8)]
            self.labels = self.labels[:int(len(self.labels) * 0.8)]
        elif stage == 3:
            self.documents = self.documents[:int(len(self.documents) * 1)]
            self.labels = self.labels[:int(len(self.labels) * 1)]
        dataset = [{'text': text, 'label': label} for text, label in zip(self.documents, self.labels)]
        dataset = self._oversample(dataset)

        self.documents = [item['text'] for item in dataset]
        self.labels = [item['label'] for item in dataset]
        for i in range(len(self.documents)):
            self.encodings.append(tokenizer(self.documents[i], truncation=True, padding='max_length'))
    def _oversample(self, data):

        type_of_each = {}

        maxim = 0
        for item in data:
            label = item["label"]
            if label not in type_of_each:
                type_of_each[label] = 1
            else:
                type_of_each[label] += 1
            if type_of_each[label] > maxim:
                maxim = type_of_each[label]

        oversample_factor = {}
        for key in type_of_each:
            oversample_factor[key] = maxim // type_of_each[key]

        samples = []
        for item in data:
            label = item["label"]
            samples += [item] * oversample_factor[label]

        random.shuffle(samples)

        print(type_of_each)
        
        type_of_each = {}
        for item in samples:
            label = item["label"]
            if label not in type_of_each:
                type_of_each[label] = 1
            else:
                type_of_each[label] += 1
                
        print(type_of_each)
        
        return samples    
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return (torch.tensor(self.encodings[index]['input_ids']), torch.tensor(self.encodings[index]['attention_mask']), torch.tensor(self.labels[index]))
    
import numpy as np
#probs = [0.3, 0.2, 0.1]  *15
#probs = [0.3, 0.4, 0.3, 0.2, 0.1]  *15
probs = []
probs = np.linspace(0, 0.15, 40)[::-1]
probs = [0.15,0.12,0.09] * 15

probs = [0.5,0.3,0.15] * 15

probs = [0.35,0.25,0.15,0.05,0.01] * 15

probs = np.linspace(0.05, 0.25, 40)

probs = [0.35,0.3,0.25] * 15
probs = [1,2,3,4,5] 
#probs = [0.25,0.2,0.15,0.1,0.05] * 15
#probs = [0.15,0.135,0.12,0.105,0.09] * 15
#for i in range(40):
#    probs.append(i/100)
#probs = probs[::-1]
train_dataloader1 = DataLoader(CustomDataset_train(data_train, train_label, probs[0], model, sentiment_dict = []), shuffle=False, batch_size=batch_size)
train_dataloader2 = DataLoader(CustomDataset_train(data_train, train_label, probs[1], model, sentiment_dict = []), shuffle=False, batch_size=batch_size)
train_dataloader3 = DataLoader(CustomDataset_train(data_train, train_label, probs[2], model, sentiment_dict = []), shuffle=False, batch_size=batch_size)

#train_dataloader4 = DataLoader(CustomDataset_train(data_train, probs[3], model, sentiment_dict = sentiwordnet_words), shuffle=True, batch_size=batch_size)
#train_dataloader5 = DataLoader(CustomDataset_train(data_train, probs[4], model, sentiment_dict = sentiwordnet_words), shuffle=True, batch_size=batch_size)

#0,1,2,9,10,11
#3,4,5,12,13,14
#6,7,8,15,16,17
import gc

#train_dataloader = DataLoader(CustomDataset(data_train), shuffle=False, batch_size=batch_size)
validation_dataloader = DataLoader(CustomDataset(data_val), shuffle=False, batch_size=batch_size)
best_f1_for_epoch = 0
epochs_without_improvement = 0
from tqdm import trange
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

steps_per_stage = [424, 848, 1272, 1696, 6380 ]  
steps_per_stage = [817 , 1634, 8170 ] 
eval_interval = 354 

from itertools import cycle
current_stage = 0
current_step = 0
total_steps = sum(steps_per_stage)

for stage_idx, stage_steps in enumerate(steps_per_stage):
    print(f"\nStarting Stage {stage_idx + 1}, Training for {stage_steps} steps\n")

    if current_stage == 0:
        train_dataloader = train_dataloader1
    elif current_stage == 1:
        train_dataloader = train_dataloader2
    elif current_stage == 2:
        train_dataloader = train_dataloader3
    elif current_stage == 3:
        train_dataloader = train_dataloader4
    elif current_stage == 4:
        train_dataloader = train_dataloader5

    train_iterator = cycle(train_dataloader) 
    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for step in trange(stage_steps, desc=f"Stage {stage_idx+1} Training"):
        batch = next(train_iterator) 
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits

        loss = loss_func(logits, b_labels.long())
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_steps += 1
        current_step += 1

        if current_step % eval_interval == 0:

            print(f"\nEvaluating at step {current_step}...\n")
            model.eval()
            val_loss = 0
            nb_val_steps = 0
            true_labels, pred_labels = [], []

            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    logits = outputs.logits
                    loss = loss_func(logits, b_labels.long())

                    val_loss += loss.item()
                    nb_val_steps += 1

                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = b_labels.cpu().numpy()

                true_labels.extend(labels)
                pred_labels.extend(preds)

            val_accuracy = accuracy_score(true_labels, pred_labels)
            val_f1 = f1_score(true_labels, pred_labels, average="micro")
            val_precision = precision_score(true_labels, pred_labels, average="micro")
            val_recall = recall_score(true_labels, pred_labels, average="micro")

            print(f'\nEvaluation at step {current_step}:')
            print(f'Validation Loss: {val_loss / nb_val_steps:.4f}')
            print(f'Accuracy: {val_accuracy:.4f}')
            print(f'F1 Score: {val_f1:.4f}')
            print(f'Precision: {val_precision:.4f}')
            print(f'Recall: {val_recall:.4f}\n')

    print(f"\nFinished Stage {stage_idx + 1}. Total steps so far: {current_step}\n")
