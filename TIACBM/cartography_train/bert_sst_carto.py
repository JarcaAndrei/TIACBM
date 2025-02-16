import torch
import numpy as np
import datasets
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW, Adamax
from torch.nn import CrossEntropyLoss

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)

epochs = 20
batch_size = 64
learning_rate = 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = Adamax(model.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()

confidence_dict = defaultdict(list)
correctness_dict = defaultdict(list)

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

        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        preds = torch.argmax(probs, dim=1)

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

class SST2Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(self.labels[idx])

if __name__ == "__main__":
    data_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))
    data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
    labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
    class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
    train_num=pickle.load(open(os.path.join(source_dir, 'data', 'train_num.'+suffix),'rb'))
    num_labels = len(labels_ref)

    texts1 = datasets.load_dataset("glue", "sst2")["train"]
    texts, labels = texts1["sentence"], texts1["label"]
    dataset = SST2Dataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train(model, dataloader, optimizer, loss_fn, device)
    
    cartography_results = compute_cartography_scores()
    sorted_indices = sorted(cartography_results.keys(), key=lambda idx: cartography_results[idx]["difficulty"])

    sorted_texts = [texts[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_dataset = {"sentence": sorted_texts, "label": sorted_labels}

    datasets.Dataset.from_dict(sorted_dataset).to_json("sorted_sst21.json")
    print("Sorted dataset saved as sorted_sst2.json")
