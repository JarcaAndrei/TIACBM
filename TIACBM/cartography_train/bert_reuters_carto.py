import torch
import numpy as np
import datasets
import pickle
import os
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from util_loss import ResampleLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 512
batch_size = 32
lr = 1e-4 
epochs = 20

source_dir = './'
prefix = 'reuters' 
suffix = 'rand123'
model_name = 'hopeitworks234'
loss_func_name = 'CBloss-ntr'
data_train = pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))
data_val = pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
labels_ref = pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq = pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
train_num = pickle.load(open(os.path.join(source_dir, 'data', 'train_num.'+suffix),'rb'))
num_labels = len(labels_ref)
from torch import nn

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', max_length=max_len)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)).to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight'] 
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=lr)
loss_fn = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                             class_freq=class_freq, train_num=train_num)

confidence_dict = defaultdict(list)
correctness_dict = defaultdict(list)

class ReutersDataset(Dataset):
    def __init__(self, data, labels_ref, tokenizer, max_length=512):
        self.texts = [item['text'] for item in data]
        self.labels = [[1 if label in item['labels'] else 0 for label in labels_ref] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(self.labels[idx], dtype=torch.float)

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    sample_idx = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        logits = model(**inputs).logits
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        conf = torch.mean(probs, dim=1)

        for i in range(len(labels)):
            confidence_dict[sample_idx].append(conf[i].item())
            correctness_dict[sample_idx].append(int(torch.all(preds[i] == labels[i])))
            sample_idx += 1

def compute_cartography_scores():
    cartography_scores = {}
    for idx in confidence_dict.keys():
        conf_values = np.array(confidence_dict[idx])
        correctness_values = np.array(correctness_dict[idx])
        confidence = np.mean(conf_values)
        variability = np.std(conf_values)
        correctness = np.mean(correctness_values)
        cartography_scores[idx] = {
            "confidence": confidence,
            "variability": variability,
            "correctness": correctness,
            "difficulty": difficulty_score(confidence, variability)
        }
    return cartography_scores

def difficulty_score(c, v):
    return 1 - c + v if c > 0.5 else 3 - c - v

dataset_train = ReutersDataset(data_train, labels_ref, tokenizer)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train(model, dataloader_train, optimizer, loss_fn, device)

cartography_results = compute_cartography_scores()
sorted_indices = sorted(cartography_results.keys(), key=lambda idx: cartography_results[idx]["difficulty"])

sorted_texts = [data_train[i]['text'] for i in sorted_indices]
sorted_labels = [data_train[i]['labels'] for i in sorted_indices]
sorted_dataset = {"text": sorted_texts, "labels": sorted_labels}

datasets.Dataset.from_dict(sorted_dataset).to_json("sorted_reuters.json")
print("Sorted dataset saved as sorted_reuters.json")
