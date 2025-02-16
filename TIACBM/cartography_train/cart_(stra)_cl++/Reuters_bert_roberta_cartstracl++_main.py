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
########## Configuration Part 1 ###########
source_dir = './'
prefix = 'reuters' 
suffix = 'rand123'
model_name = 'hopeitworks234'
loss_func_name = 'CBloss-ntr'

data_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))

file_path = r"sorted_reuters_roberta.json"
data_train = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f: 
        data = json.loads(line.strip())  
        data_train.append({"text": data["text"], "labels": data["labels"]})
data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
train_num=pickle.load(open(os.path.join(source_dir, 'data', 'train_num.'+suffix),'rb'))
num_labels = len(labels_ref)
from sklearn.datasets import fetch_20newsgroups
#data_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))['data']
#data_val = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))['data']
#labels_ref = 
max_len = 512
lr = 5e-5 #roberta
lr = 1e-4 #bert
epochs = 30
batch_size = 32

########## set up ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', max_len=max_len)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels=num_labels)).to(device)

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
#spacy.require_gpu()
nlp = spacy.load("en_core_web_lg") #accuracy

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
import pickle

from collections import defaultdict
class CustomDataset_train(Dataset):
    def __init__(self, documents, stage, model, fractions):
        self.documents = documents
        self.device = 'cuda'
        self.model = model
        self.stage = stage
        self.fractions = fractions
        self.labels = []
        self.encodings = []
        label_groups = {}
        for doc in self.documents:
            label_tuple = tuple(sorted(doc['labels'])) 
            if label_tuple not in label_groups:
                label_groups[label_tuple] = []
            label_groups[label_tuple].append(doc)
        
        selected_docs = []
        for label_tuple, docs in label_groups.items():
            num_samples = int(len(docs) * self.fractions[self.stage])
            selected_docs.extend(docs[:num_samples])
        
        self.documents = selected_docs

        #if mask_prob == 1:
        #    fraction = 0.6
        #elif mask_prob == 2:
        #    fraction = 0.8
        #else:  # mask_prob == 3
        #    fraction = 1.0

        #self.documents = []
        #for label, items in label_dict.items():
        #    self.documents.extend(items[:int(len(items) * fraction)])
        #if mask_prob == 1:
        #    self.documents = self.documents[:int(len(self.documents) * 0.6)]
        #elif mask_prob == 2:
        #    self.documents = self.documents[:int(len(self.documents) * 0.8)]
        #elif mask_prob == 3:
        #    self.documents = self.documents[:int(len(self.documents) * 1)]
        '''
        self.fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.train_stage = mask_prob
        label_dict = defaultdict(list)

        for entry in self.documents:
            text, labels = entry["text"], entry["labels"]
            for label in labels:  # Since labels are a list, iterate through them
                label_dict[label].append(entry)  

        filtered_documents = []
        for label, items in label_dict.items():
            num_samples = int(len(items) * self.fractions[self.train_stage])
            filtered_documents.extend(items[:num_samples]) 

        seen = set()
        self.documents = []
        for doc in filtered_documents:
            doc_tuple = (doc["text"], tuple(doc["labels"]))
            if doc_tuple not in seen:
                seen.add(doc_tuple)
                self.documents.append(doc)
        print(f"Total filtered samples: {len(self.documents)}") 
        '''

        #self.docs = docs
        #self.documents = self._oversample(self.documents)
        for i in range(len(self.documents)):
            self.labels.append([1 if x in self.documents[i]['labels'] else 0 for x in labels_ref])
            self.encodings.append(tokenizer(self.documents[i]['text'], truncation=True, padding='max_length'))
    def __len__(self):
        return len(self.documents)
    import random
    from collections import Counter

    def _oversample(self, data):
        """
        Oversamples data to balance multi-label classification.
        
        Args:
            data (list): List of dictionaries where each entry has a "label" key with a list of labels.
        
        Returns:
            list: Oversampled dataset.
        """
        label_counts = Counter()

        # Count occurrences for each label
        for item in data:
            for label in item["labels"]:  # Assuming label is a list of labels
                label_counts[label] += 1

        # Find the maximum occurrence of any label
        max_label_count = max(label_counts.values())

        # Compute oversampling factors for each label
        oversample_factors = {label: max_label_count / count for label, count in label_counts.items()}

        # Oversampling based on the highest label factor
        oversampled_data = []
        for item in data:
            labels = item["labels"]
            
            # Determine how many times this sample should be repeated
            repeat_factor = max(oversample_factors[label] for label in labels)
            
            oversampled_data.extend([item] * int(repeat_factor))  # Duplicate samples

        # Shuffle to avoid bias
        random.shuffle(oversampled_data)

        # Debugging: Print label distribution before and after oversampling
        print("Original Label Distribution:", dict(label_counts))
        
        new_label_counts = Counter()
        for item in oversampled_data:
            for label in item["labels"]:
                new_label_counts[label] += 1
                
        print("New Label Distribution:", dict(new_label_counts))
        
        return oversampled_data

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

probs = np.linspace(0.05, 0.3, 40)
#probs = [0.3,0.25,0.2] * 15
#probs = [0.15, 0.22, 0.3]* 15
#probs = [0.15,0.15,0.15] * 15

#probs = [0.25,0.2,0.15,0.1,0.05] * 15
#probs = [0.15,0.135,0.12,0.105,0.09] * 15
#for i in range(40):
#    probs.append(i/100)
#probs = probs[::-1]
import gc
from itertools import cycle
import numpy as np
import torch
from tqdm import trange
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

STAGE_STEPS  = [240, 510, 750, 1020, 3810]
#STAGE_STEPS = [488 ,976 ,4882 ]
EVAL_INTERVAL = 212   # Evaluate every 424 steps
probs = [0.2,0.4,0.6,0.8,1]
# Define dataloaders for each stage
train_dataloader1 = DataLoader(CustomDataset_train(data_train, 0, model, probs), shuffle=True, batch_size=batch_size)
train_dataloader2 = DataLoader(CustomDataset_train(data_train, 1, model, probs), shuffle=True, batch_size=batch_size)
train_dataloader3 = DataLoader(CustomDataset_train(data_train, 2, model, probs), shuffle=True, batch_size=batch_size)
train_dataloader4 = DataLoader(CustomDataset_train(data_train, 3, model, probs), shuffle=True, batch_size=batch_size)
train_dataloader5 = DataLoader(CustomDataset_train(data_train, 4, model, probs), shuffle=True, batch_size=batch_size)

# Validation dataloaderf 
validation_dataloader = DataLoader(CustomDataset(data_val), shuffle=False, batch_size=batch_size)

# Optimizer and tracking variables
best_f1_for_epoch = 0
epochs_without_improvement = 0
global_step = 0

# Cycle through stages
for stage_idx, stage_steps in enumerate(STAGE_STEPS):
    print(f"\nStarting Stage {stage_idx + 1}, Training for {stage_steps} steps\n")
    
    # Select the appropriate dataloader
    if stage_idx == 0:
        train_dataloader = train_dataloader1
    elif stage_idx == 1:
        train_dataloader = train_dataloader2
    elif stage_idx == 2:
        train_dataloader = train_dataloader3
    elif stage_idx == 3:
        train_dataloader = train_dataloader4
    elif stage_idx == 4:
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
        logits = outputs[0]
        loss = loss_func(logits.view(-1, num_labels), b_labels.type_as(logits).view(-1, num_labels))
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_steps += 1
        global_step += 1

        if global_step % EVAL_INTERVAL == 0:
            print(f"\nEvaluating at step {global_step}...\n")
            
            model.eval()
            val_loss = 0
            nb_val_steps = 0
            true_labels, pred_labels = [], []

            with torch.no_grad():
                for batch in validation_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    b_logit_pred = outs[0]
                    pred_label = torch.sigmoid(b_logit_pred)
                    loss = loss_func(b_logit_pred.view(-1, num_labels), b_labels.type_as(b_logit_pred).view(-1, num_labels))
                    val_loss += loss.item()
                    nb_val_steps += 1

                    true_labels.append(b_labels.cpu().numpy())
                    pred_labels.append(pred_label.cpu().numpy())
            val_loss /= nb_val_steps
            print(f"Validation loss: {val_loss:.4f}")

            true_labels = np.concatenate(true_labels, axis=0)
            pred_labels = np.concatenate(pred_labels, axis=0)

            threshold = 0.5
            true_bools = true_labels == 1
            pred_bools = pred_labels > threshold
            val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro')
            val_precision_accuracy = precision_score(true_bools, pred_bools, average='micro')
            val_recall_accuracy = recall_score(true_bools, pred_bools, average='micro')

            print(f'F1 Validation Accuracy: {val_f1_accuracy:.4f}')
            print(f'Precision Validation Accuracy: {val_precision_accuracy:.4f}')
            print(f'Recall Validation Accuracy: {val_recall_accuracy:.4f}')

            val_auc_score = roc_auc_score(true_bools, pred_labels, average='micro')
            print(f'AUC Validation: {val_auc_score:.4f}')

            best_med_th = 0.5
            micro_thresholds = (np.arange(-10, 11) / 100) + best_med_th
            f1_results = [
                f1_score(true_bools, pred_labels > th, average='micro')
                for th in micro_thresholds
            ]
            best_f1_idx = np.argmax(f1_results)

            print(f'Best Threshold: {micro_thresholds[best_f1_idx]:.4f}')
            print(f'Test F1 Accuracy: {f1_results[best_f1_idx]:.4f}')

            if f1_results[best_f1_idx] > (best_f1_for_epoch * 0.995):
                best_f1_for_epoch = f1_results[best_f1_idx]
                epochs_without_improvement = 0
                model_dir = os.path.join(source_dir, 'models')

                print(f'Model saved at step {global_step}')
            else:
                epochs_without_improvement += 1

    print(f"\nFinished Stage {stage_idx + 1}. Total steps so far: {global_step}\n")

