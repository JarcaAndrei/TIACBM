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
model_name = 'hopeitworks234'
loss_func_name = 'CBloss-ntr'

data_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))
data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
train_num=pickle.load(open(os.path.join(source_dir, 'data', 'train_num.'+suffix),'rb'))
num_labels = len(labels_ref)
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

        
from torch.utils.data import Dataset, DataLoader

def preprocess_function(docu):
    labels = [1 if x in docu['labels'] else 0 for x in labels_ref] 
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

def extract_synset_from_line(line):
    parts = line.split('\t')
    pos = parts[0]  
    synset_id = parts[1] 
    synset = f"{pos}.{synset_id}" 
    return synset
def load_sentiwordnet(file_path):
    sentiwordnet_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  
            parts = line.split('\t')
            if len(parts) < 6:
                continue 
            synset = extract_synset_from_line(line)
            try:
                pos_score = float(parts[2])
            except:
                pos_score = 0.
            try:
                neg_score = float(parts[3])
            except:
                neg_score = 0.
            sentiwordnet_data[synset] = {'pos_score': pos_score, 'neg_score': neg_score}
    return sentiwordnet_data

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

def date_to_words(date_str):
    date_obj = datetime.datetime.strptime(date_str, "%d-%b-%Y")
    day_text = num2words(date_obj.day, ordinal=True)  
    month_text = date_obj.strftime("%B") 
    year_text = num2words(date_obj.year) 

    return f"{day_text} of {month_text}, {year_text}"
def time_to_words(time_str):
    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S.%f").time()
    
    hours = num2words(time_obj.hour)
    minutes = num2words(time_obj.minute)
    seconds, millis = divmod(time_obj.second + time_obj.microsecond / 1_000_000, 1)
    
    seconds_text = num2words(int(seconds))
    millis_text = num2words(int(millis * 100)) 
    time_text = f"{hours} hours, {minutes} minutes, and {seconds_text} point {millis_text} seconds"
    
    return time_text
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

def clean_text(input_text):
    cleaned_text = re.sub(r"&[a-zA-Z0-9#]+;", "", input_text)
    
    cleaned_text = re.sub(r"<.*?>", "", cleaned_text)
    cleaned_text = re.sub(r"[(){}\[\]]", " ", cleaned_text)  
    return cleaned_text
def preprocess_text(text):
    doc = nlp(clean_text(text.lower())) 
    tokens = []
    itt = 0
    print(len(doc))
    for token in doc:
        itt+=1
        if token.pos_ in REMOVE_POS:
            continue  
        if token.pos_ == "NUM":  
            print(token.text, itt)
            token = preprocess_num_token(token.text)
            print(token)
            if token == '':
                continue
            tokens.append(token) 
        elif token.pos_ == "SYM":
            tokens.append(token.text)  
        else:
            tokens.append(token.lemma_) 
    return " ".join(tokens)
def preprocess_text1(text):
    doc = nlp(clean_text(text.lower())) 
    tokens = []
    itt = 0
    for token in doc:
        itt+=1
        if token.pos_ in REMOVE_POS:
            continue  
        if token.pos_ == "NUM":  
            continue
            token = preprocess_num_token(token.text)
            tokens.append(token)
            
        elif token.pos_ == 'INTJ':
            continue
            tokens.append(token.text)
        elif token.pos_ == "SYM":
            continue
            tokens.append(token.text)  
        else:
            tokens.append(token.lemma_) 

    return " ".join(tokens)

docs = []
for i in range(len(data_train)):
    docs.append(nlp(data_train[i]['text'].lower())) #

import pickle

class CustomDataset_train(Dataset):
    def __init__(self, documents, mask_prob, model, sentiment_dict=None):
        self.documents = documents
        self.mask_prob = mask_prob
        self.device = 'cuda'
        self.labels = []
        self.sentiment_dict = sentiment_dict
        self.model = model
        self.encodings = []
        self.docs = docs
        for i in range(len(self.documents)):
            self.labels.append([1 if x in self.documents[i]['labels'] else 0 for x in labels_ref])
            self.encodings.append(tokenizer(self.documents[i]['text'], truncation=True, padding='max_length'))
        if self.mask_prob != 0:
            for i in tqdm(range(len(self.encodings)), desc="Masking tokens"):
                self.encodings[i]['input_ids'] = self.mask_content_words_with_attention(self.encodings[i]['input_ids'], self.encodings[i]['attention_mask'], self.docs[i])
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return (torch.tensor(self.encodings[index]['input_ids']), torch.tensor(self.encodings[index]['attention_mask']), torch.tensor(self.labels[index]))
    
    
    def mask_content_words_with_attention(self, input_ids, attention_mask, doc):
        
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        attention_mask_tensor = torch.tensor(attention_mask).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, output_attentions=True)
            attentions = outputs.attentions 
        avg_attention = torch.mean(torch.stack(attentions), dim=(0, 2))
        token_importance = torch.mean(avg_attention, dim=-1).squeeze(0) 
        token_importance = (token_importance - token_importance.min()) / (token_importance.max() - token_importance.min())
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        words = tokenizer.convert_tokens_to_string(tokens).split()
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
        word_indices = {}
        poses_scores = {'NOUN': [], 'VERB': [], 'ADJ': [], 'ADV': [], 'PROPN': []}
        imps = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PROPN': 0}
        
        struc_for_poses={}
        for token in doc:
            word = token.text
            pos = token.pos_ 
            if pos in content_pos:
                blas = get_word_token_indices(tokens, word, tokenizer)
                word_indices[word] = blas
                struc_for_poses[word] = (pos, blas)
        all_indices = [(index, 1) for word, indices in word_indices.items() for index in indices]
        
        masked_tokens = tokens.copy()

        def find_pos(token_idx):
            return index_to_pos.get(token_idx, None)

        for idx, imps in all_indices:
            for idxx in idx:
                if masked_tokens[idxx] not in tokenizer.all_special_tokens:

                    adjusted_mask_prob = self.mask_prob * ((1 + token_importance[idxx].item()))

                    if random.random() < adjusted_mask_prob:
                        masked_tokens[idxx] = '[MASK]'
        masked_input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
        return masked_input_ids
import numpy as np
#probs = [0.3, 0.2, 0.1]  *15
#probs = [0.3, 0.4, 0.3, 0.2, 0.1]  *15
probs = []
probs = np.linspace(0, 0.15, 40)[::-1]
probs = [0.15,0.12,0.09] * 15

probs = [0.5,0.3,0.15] * 15

probs = [0.35,0.25,0.15,0.05,0.01] * 15

probs = np.linspace(0.05, 0.3, 40)
probs = np.linspace(0.15, 0, 30)
probs = [0.15] *15
probs = [0.09,0.12,0.15]*15
probs = [0.15, 0.12, 0.09] *15
#probs = [0,0,0]
#probs = [0.3,0.25,0.2] * 15
#probs = [0.15, 0.22, 0.3]* 15
#probs = [0.15,0.15,0.15] * 15

#probs = [0.25,0.2,0.15,0.1,0.05] * 15
#probs = [0.15,0.135,0.12,0.105,0.09] * 15
#probs = [0.15,0.12,0.09] * 15
#probs = [0.35, 0, 0.25] * 15
#for i in range(40):    
#    probs.append(i/100)
#probs = probs[::-1]
sentiwordnet_words = []
train_dataloader1 = DataLoader(CustomDataset_train(data_train, probs[0], model, sentiment_dict = sentiwordnet_words), shuffle=True, batch_size=batch_size)
train_dataloader2 = DataLoader(CustomDataset_train(data_train, probs[1], model, sentiment_dict = sentiwordnet_words), shuffle=True, batch_size=batch_size)
train_dataloader3 = DataLoader(CustomDataset_train(data_train, probs[2], model, sentiment_dict = sentiwordnet_words), shuffle=True, batch_size=batch_size)
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

for epoch in trange(epochs, desc="Epoch"):
    #train_dataloader = DataLoader(CustomDataset_train(data_train, probs[epoch], model, sentiment_dict = sentiwordnet_words), shuffle=True, batch_size=batch_size)

    #if epoch != 0:
    #    train_dataloader = DataLoader(CustomDataset_train(data_train, probs[epoch], model, sentiment_dict = sentiwordnet_words), shuffle=True, batch_size=batch_size)
    if epoch % 3 == 0:
        train_dataloader = train_dataloader1
    elif epoch % 3 == 1:
        train_dataloader = train_dataloader2
    elif epoch % 3 == 2:
        train_dataloader = train_dataloader3
    #elif epoch % 5 == 3:
    #    train_dataloader = train_dataloader4
    #elif epoch % 5 == 4:
    #    train_dataloader = train_dataloader5
    #if (epoch // 3) % 3 == 0:
    #    train_dataloader = train_dataloader1
    #elif (epoch // 3) % 3 == 1:
    #    train_dataloader = train_dataloader2
    #else:
    #    train_dataloader = train_dataloader3
    # Training
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
  
    for _, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()

        outputs = model(b_input_ids,  attention_mask=b_input_mask)
        logits = outputs[0]
        loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels))
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    model.eval()
    val_loss = 0
    nb_val_steps = 0
    true_labels,pred_labels = [],[]
    
    for _, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outs = model(b_input_ids,  attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
            loss = loss_func(b_logit_pred.view(-1,num_labels),b_labels.type_as(b_logit_pred).view(-1,num_labels))
            val_loss += loss.item()
            nb_val_steps += 1
    
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        true_labels.append(b_labels)
        pred_labels.append(pred_label)
    
    print("Validation loss: {}".format(val_loss/nb_val_steps))

    true_labels = [item for sublist in true_labels for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]

    threshold = 0.5
    true_bools = [tl==1 for tl in true_labels]
    pred_bools = [pl>threshold for pl in pred_labels]
    val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
    val_precision_accuracy = precision_score(true_bools, pred_bools,average='micro')
    val_recall_accuracy = recall_score(true_bools, pred_bools,average='micro')
    
    print('F1 Validation Accuracy: ', val_f1_accuracy)
    print('Precision Validation Accuracy: ', val_precision_accuracy)
    print('Recall Validation Accuracy: ', val_recall_accuracy)

    val_auc_score = roc_auc_score(true_bools, pred_labels, average='micro')
    print('AUC Validation: ', val_auc_score)
    
    best_med_th = 0.5
    micro_thresholds = (np.array(range(-10,11))/100)+best_med_th
    f1_results, prec_results, recall_results = [], [], []
    for th in micro_thresholds:
        pred_bools = [pl>th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
        test_precision_accuracy = precision_score(true_bools, pred_bools,average='micro')
        test_recall_accuracy = recall_score(true_bools, pred_bools,average='micro')
        f1_results.append(test_f1_accuracy)
        prec_results.append(test_precision_accuracy)
        recall_results.append(test_recall_accuracy)

    best_f1_idx = np.argmax(f1_results) 

    print('Best Threshold: ', micro_thresholds[best_f1_idx])
    print('Test F1 Accuracy: ', f1_results[best_f1_idx])
  
    if f1_results[best_f1_idx] > (best_f1_for_epoch * 0.995):
        best_f1_for_epoch = f1_results[best_f1_idx]
        epochs_without_improvement = 0
        model_dir = os.path.join(source_dir, 'models')
        for fname in os.listdir(model_dir):
            if fname.startswith('_'.join([prefix,model_name,loss_func_name,suffix])):
                os.remove(os.path.join(model_dir, fname))
        torch.save(model.state_dict(), os.path.join(model_dir, '_'.join([prefix,model_name,loss_func_name,suffix,'epoch'])+str(epoch+1)+'para'))
    else:
        epochs_without_improvement += 1
    
