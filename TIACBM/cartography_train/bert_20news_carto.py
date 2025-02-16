import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset as HFDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 512
batch_size = 32
lr = 1e-4
epochs = 20

data_train = fetch_20newsgroups(subset="train")
data_val = fetch_20newsgroups(subset="test")
train_texts, train_labels = data_train['data'], data_train['target']
val_texts, val_labels = data_val['data'], data_val['target']
num_labels = len(set(train_labels))

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', max_length=max_len)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)

optimizer = AdamW(model.parameters(), lr=lr)
loss_fn = CrossEntropyLoss()

confidence_dict = defaultdict(list)
correctness_dict = defaultdict(list)

class NewsGroupsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    sample_idx = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(**inputs).logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values

        for i in range(len(labels)):
            confidence_dict[sample_idx].append(conf[i].item())
            correctness_dict[sample_idx].append(int(preds[i] == labels[i]))
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

dataset_train = NewsGroupsDataset(train_texts, train_labels, tokenizer)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train(model, dataloader_train, optimizer, loss_fn, device)

cartography_results = compute_cartography_scores()
sorted_indices = sorted(cartography_results.keys(), key=lambda idx: cartography_results[idx]["difficulty"])

sorted_texts = [train_texts[i] for i in sorted_indices]
sorted_labels = [train_labels[i] for i in sorted_indices]
sorted_dataset = {"text": sorted_texts, "labels": sorted_labels}

HFDataset.from_dict(sorted_dataset).to_json("sorted_20news.json")
print("Sorted dataset saved as sorted_20news.json")
