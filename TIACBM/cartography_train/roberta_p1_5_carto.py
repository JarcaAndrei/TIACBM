import os
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report
DATA_PATH = "path/to/problem00001_or_problem00005"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-5
LR = 5e-5
MAX_LEN = 512
MODEL_NAME = "FacebookAI/roberta-base"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

ground_truth = load_json(os.path.join(DATA_PATH, "ground-truth.json"))
problem_info = load_json(os.path.join(DATA_PATH, "problem-info.json"))
candidate_authors = [author["author-name"] for author in problem_info["candidate-authors"]]

ground_truth_dict = {entry["unknown-text"]: entry["true-author"] for entry in ground_truth["ground_truth"]}

author_to_label = {author: idx for idx, author in enumerate(candidate_authors)}
label_to_author = {idx: author for author, idx in author_to_label.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(candidate_authors)).to('cuda')

optimizer = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

confidence_dict = defaultdict(list)
correctness_dict = defaultdict(list)

class PAN19Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for author in candidate_authors:
            author_dir = os.path.join(data_path, author)
            if not os.path.exists(author_dir):
                continue
            for file in os.listdir(author_dir):
                file_path = os.path.join(author_dir, file)
                text = read_txt(file_path)
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                self.samples.extend(chunks)
                self.labels.extend([author_to_label[author]] * len(chunks))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.samples[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
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

dataset_train = PAN19Dataset(DATA_PATH, tokenizer)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda"
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train(model, dataloader_train, optimizer, criterion, device)

cartography_results = compute_cartography_scores()
sorted_indices = sorted(cartography_results.keys(), key=lambda idx: cartography_results[idx]["difficulty"])

sorted_texts = [dataset_train.samples[i] for i in sorted_indices]
sorted_labels = [dataset_train.labels[i] for i in sorted_indices]
sorted_dataset = {"text": sorted_texts, "labels": sorted_labels}

HFDataset.from_dict(sorted_dataset).to_json("sorted_pan19_1_roberta.json")
print("Sorted dataset saved as sorted_pan19_1_roberta.json")