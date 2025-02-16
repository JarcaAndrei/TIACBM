import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
DATA_PATH = "path/to/problem0005"
BATCH_SIZE = 8
EPOCHS = 30
LR = 5e-5
MAX_LEN = 512
MODEL_NAME = "FacebookAI/roberta-base"

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
#candidate_authors.append("<UNK>")
ground_truth_dict = {entry["unknown-text"]: entry["true-author"] for entry in ground_truth["ground_truth"]}
author_to_label = {author: idx for idx, author in enumerate(candidate_authors)}
label_to_author = {idx: author for author, idx in author_to_label.items()}

def get_word_token_indices(tokens, word, tokenizer):
    word_tokens = tokenizer.tokenize(word)
    word_len = len(word_tokens)
    indices = []
    for i in range(len(tokens) - word_len + 1):
        if tokens[i:i + word_len] == word_tokens:
            indices.append(list(range(i, i + word_len)))
    return indices
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
import spacy
nlp = spacy.load("en_core_web_lg")
import random
class AuthorshipDataset(Dataset):
    def __init__(self, data_path, probs, tokenizer, max_len, model):
        self.samples = []
        self.model = model
        self.device = 'cuda'
        self.mask_prob = probs
        for author in candidate_authors:
            author_folder = os.path.join(data_path, author)
            if os.path.exists(author_folder):
                for file in os.listdir(author_folder):
                    file_path = os.path.join(author_folder, file)
                    text = read_txt(file_path)
                    chunks = self.tokenize_text(text, tokenizer, max_len)
                    self.samples.extend([(chunk, author_to_label[author]) for chunk in chunks])
        self.docs = []
        for i in range(len(self.samples)):
            self.docs.append(nlp(self.samples[i][0]))
    def tokenize_text(self, text, tokenizer, max_len):
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        return [tokenizer.decode(tokens[i:i + max_len]) for i in range(0, len(tokens), max_len // 2)]
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
        

        content_pos = {'ADP', 'DET', 'CONJ', 'CCONJ',  'SCONJ',  'SYM', 'PART', 'PUNCT'}
        word_indices = {}
        poses_scores = {'ADP': [], 'DET': [], 'CONJ':[],   'CCONJ': [],  'SCONJ': [],  'SYM': [], 'PART':[], 'PUNCT':[]}
        imps = {'ADP':0, 'DET': 0, 'CONJ':0,  'CCONJ': 0, 'SCONJ': 0, 'SYM': 0, 'PART':0, 'PUNCT':0}
        
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
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids, attention_mask = inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)
        if self.mask_prob != 0:
            input_ids = self.mask_content_words_with_attention(input_ids, attention_mask, self.docs[idx])
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask), "label": torch.tensor(label)}

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
probs = [0.35,0.3,0.25] 
probs = [0,0,0] 
#train_dataset1 = AuthorshipDataset(DATA_PATH, probs[0], tokenizer, MAX_LEN, model)
#train_dataset2 = AuthorshipDataset(DATA_PATH,probs[1], tokenizer, MAX_LEN, model)
#train_dataset3 = AuthorshipDataset(DATA_PATH,probs[2], tokenizer, MAX_LEN, model)

#train_loader1 = DataLoader(AuthorshipDataset(DATA_PATH, probs[0], tokenizer, MAX_LEN, model), batch_size=BATCH_SIZE, shuffle=True)
#train_loader2 = DataLoader(train_dataset2, batch_size=BATCH_SIZE, shuffle=True)
#train_loader3 = DataLoader(train_dataset3, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TestDataset(os.path.join(DATA_PATH, "unknown"), tokenizer, MAX_LEN)
import numpy as np
#arr = [0.3]*30
arr = np.linspace(0.3,0,30)
#arr = [0,0,0,0.35,0.3,0.25,]
#93 80 on 35 3 25
def train():
    model.train()
    #probs = [0.3,0.3,0.3]  
    #probs = [0.3,0,0.25]
       #0.79
       #821
    #probs = [0.35, 0, 0.25]
    probs = [0,0,0]
    #probs = [0.3, 0.3, 0.3]
    #probs = [0.35,0,0.25]

    for epoch in range(EPOCHS):
        if epoch==3:

            #probs = [0.15,0,0.1]  
        #    probs = [0.35,0,0.25]  
            #probs = [0.25,0,0.35]
            #probs = [0.25,0,0.35]
            probs = [0.3,0,0.25]
        #train_loader = DataLoader(AuthorshipDataset(DATA_PATH, arr[epoch], tokenizer, MAX_LEN, model), batch_size=BATCH_SIZE, shuffle=True)

        if epoch % 3 == 0:
            train_loader = DataLoader(AuthorshipDataset(DATA_PATH, probs[0], tokenizer, MAX_LEN, model), batch_size=BATCH_SIZE, shuffle=True)
        elif epoch % 3 == 1:
            train_loader = DataLoader(AuthorshipDataset(DATA_PATH, probs[1], tokenizer, MAX_LEN, model), batch_size=BATCH_SIZE, shuffle=True)
        elif epoch % 3 == 2:
            train_loader = DataLoader(AuthorshipDataset(DATA_PATH, probs[2], tokenizer, MAX_LEN, model), batch_size=BATCH_SIZE, shuffle=True)

        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to('cuda'), batch["attention_mask"].to('cuda'), batch["label"].to('cuda')
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}")
        evaluate()

def evaluate():
    model.eval()
    y_true, y_pred = [], []
    test_dataset = TestDataset(os.path.join(DATA_PATH, problem_info["unknown-folder"]), tokenizer, MAX_LEN)
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
