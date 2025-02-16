import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import itertools

DATA_PATH = "path/to/problem00001_or_problem00005"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-5
LR = 5e-5
MAX_LEN = 512
MODEL_NAME = "FacebookAI/roberta-base"
STAGES = 3
STEPS_PER_STAGE = [84, 168, 252, 336, 1260]  
STEPS_PER_STAGE = [162, 324, 1614] 
EVAL_INTERVAL = 70 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

class TestDataset(Dataset):
    def __init__(self, test_path, tokenizer, max_len):
        self.samples = []
        for file in os.listdir(test_path):
            file_path = os.path.join(test_path, file)
            text = read_txt(file_path)
            chunks = self.tokenize_text(text, tokenizer, max_len)
            self.samples.append((file, chunks))
    
    def tokenize_text(self, text, tokenizer, max_len):
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        return [tokenizer.decode(tokens[i:i + max_len]) for i in range(0, len(tokens), max_len // 2)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
import random
class AuthorshipDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, stage):
        self.samples = []
        sentences, labels = [], []
        with open(os.path.join(data_path, "sorted_pan19_1_roberta.json"), "r", encoding="utf-8") as f:
            for line in f: 
                data = json.loads(line.strip())  
                sentences.append(data["text"])
                labels.append(data["labels"])
        self.fractions = [0.2,0.4,0.6,0.8,1]
        self.fractions = [0.6,0.8,1]
        self.train_stage = stage

        self.samples = [(text,label) for text, label in zip(sentences,labels)]
        '''
        label_0 = [item for item in data if item['label'] == 0]
        label_1 = [item for item in data if item['label'] == 1]
        label_2 = [item for item in data if item['label'] == 2]
        label_3 = [item for item in data if item['label'] == 3]
        label_4 = [item for item in data if item['label'] == 4]
        label_5 = [item for item in data if item['label'] == 5]
        label_6 = [item for item in data if item['label'] == 6]
        label_7 = [item for item in data if item['label'] == 7]
        label_8 = [item for item in data if item['label'] == 8]
        
        label_0 = label_0[:int(len(label_0)*self.fractions[self.train_stage])]
        label_1 = label_1[:int(len(label_1)*self.fractions[self.train_stage])]
        label_2 = label_2[:int(len(label_2)*self.fractions[self.train_stage])]
        label_3 = label_3[:int(len(label_3)*self.fractions[self.train_stage])]
        label_4 = label_4[:int(len(label_4)*self.fractions[self.train_stage])]
        label_5 = label_5[:int(len(label_5)*self.fractions[self.train_stage])]
        label_6 = label_6[:int(len(label_6)*self.fractions[self.train_stage])]
        label_7 = label_7[:int(len(label_7)*self.fractions[self.train_stage])]
        label_8 = label_8[:int(len(label_8)*self.fractions[self.train_stage])]
        
        train_set = label_0 + label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8
        '''
        if stage == 1:
            self.samples = self.samples[:int(len(self.samples) * 0.6)]
        elif stage == 2:
            self.samples = self.samples[:int(len(self.samples) * 0.8)]
        elif stage == 3:
            self.samples = self.samples[:int(len(self.samples) * 1)]
        #dataset = [{'text': text, 'label': label} for text, label in self.samples]
        #self.samples = self._oversample(self.samples )
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
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids, attention_mask = inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": torch.tensor(label)}

class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, output_attentions=True)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
        return outputs.logits, outputs.attentions if output_attentions else None

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(candidate_authors)).to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

def train():
    model.train()
    step = 0
    for stage in range(STAGES):
        train_loader = DataLoader(AuthorshipDataset(DATA_PATH, tokenizer, MAX_LEN, stage), batch_size=BATCH_SIZE, shuffle=True)
        train_iter = itertools.cycle(train_loader)
        stage_steps = STEPS_PER_STAGE[stage]

        while step < sum(STEPS_PER_STAGE[:stage + 1]):  
            batch = next(train_iter)
            input_ids, attention_mask, labels = batch["input_ids"].to('cuda'), batch["attention_mask"].to('cuda'), batch["label"].to('cuda')
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels.long())
            loss.backward()
            optimizer.step()
            step += 1

            if step % EVAL_INTERVAL == 0:
                print(f"Stage {stage+1}, Step {step}: Evaluating...")
                evaluate()

        print(f"Stage {stage+1} completed. Moving to the next stage.")


def evaluate():
    model.eval()
    y_true, y_pred = [], []
    test_dataset = TestDataset(os.path.join(DATA_PATH, "unknown"), tokenizer, MAX_LEN)
    with torch.no_grad():
        for filename, chunks in test_dataset:
            if ground_truth_dict[filename] == "<UNK>":
                continue 
            chunk_preds = []
            for chunk in chunks:
                inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
                input_ids, attention_mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
                logits = model(input_ids, attention_mask).logits
                chunk_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            final_pred = np.bincount(np.concatenate(chunk_preds)).argmax()
            y_pred.append(final_pred)
            y_true.append(author_to_label[ground_truth_dict[filename]])
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=candidate_authors, digits=3))

train()
