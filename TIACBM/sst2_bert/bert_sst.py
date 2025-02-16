import pandas as pd
import datasets
from dataloaders import get_train_val_loaders_sstdata
from abc import ABC

from torch import optim
from transformers import BertForSequenceClassification,BertForMaskedLM
from trainer_bert import TrainerBert
import argparse
import torch
from transformers import AutoTokenizer, BertTokenizerFast
from nltk.corpus import wordnet as wn
from functools import partial
from datautils import  BertDataProcessor
import numpy as np

parser = argparse.ArgumentParser()

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.num_classes = 2
args.num_epochs = 15
args.initial_learning_rate = 5e-5
args.model_name = '...'
def extract_synset_from_line(line):
    parts = line.split('\t')
    pos = parts[0]  # Part of speech
    synset_id = parts[1]  # Synset ID
    synset = f"{pos}.{synset_id}"  # Create synset identifier
    return synset
def load_sentiwordnet(file_path):
    sentiwordnet_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comments
            parts = line.split('\t')
            if len(parts) < 6:
                continue  # Skip malformed lines
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

# Example usage

def add_optimizer_params_lr_scale(trainer, args):
    pass
def build_optimizer_bert(model, args):
    
    return optim.Adamax(model.parameters(), lr=args.initial_learning_rate)

if __name__ == '__main__':
    sentiwordnet_words = load_sentiwordnet('SentiWordNet_3.0.0.txt')

    #94.61 +- 0.08
    #94.1 +- 0.2
    #93.94 +- 0.15
    #93.38 +- 0.14
    #mlm_model = BertForMaskedLM.from_pretrained("bert-large-uncased")
    
    #mlm_model.load_state_dict(torch.load("./output/bert_pretrained49999.pth"))


    model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
    

	#model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
    probs = []
	#best 15 -2 inverse 1+
	#best 15 -2 verse 2- per token
	#best 15 12 09 verse 2- per tok
    for i in range(15,0,-2):
        probs.append(i/100)
    #probs = [0]*40
    probs = [0.15, 0.12, 0.09]  *5
    probs = [0.09, 0.12, 0.15]  *5
    probs = np.linspace(0, 0.15,15)
    import numpy as np
    #probs = np.linspace(0.15,0,15)
    #probs = [0.09, 0.12, 0.15]  * 15
    #probs = [0.2, 0.15, 0.1] * 5
	#probs = [0.09, 0.12, 0.15] * 5
	#probs = [0.15]*15
	#probs = [0.18, 0.16, 0.14, 0.12, 0.1] * 3
	#probs = [0]*15
	#probs = [0.1,0.2,0.3,0.4,0.5] * 3
	#probs = [0] * 15

    #probs = [0.2, 0.15, 0.1] * 5
    #probs = [0.5, 0.4, 0.3, 0.2, 0.1] * 20

    probs = [0.15, 0.12, 0.09]  *5
    print(probs)	
    train, test = get_train_val_loaders_sstdata(probs = probs, epoch = 0, sentiment_dict = sentiwordnet_words)

    trainer = TrainerBert(model, train, test, add_optimizer_params_lr_scale, args,build_optimizer_bert, probs, sentiment_dict = sentiwordnet_words)

    trainer.train()
