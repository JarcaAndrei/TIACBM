import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from test import test_bert, test_bert_20news, test_roberta
from transformers import AutoTokenizer
from dataloaders import get_train_val_loaders_sstdata
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_20newsgroups
class TrainerRoberta:
    def __init__(self, model, train_loader, val_loader,add_optimizer_params_lr, args,
                 build_optimizer,probs, sentiment_dict):
        self.args = args
        self.model = model
        self.optimizer = build_optimizer(model, args)
        add_optimizer_params_lr(self, args)
        self.ce_loss = F.cross_entropy
        self.sentiment_dict = sentiment_dict
        self.model = self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.probabilities = probs

    def train(self):
        print('Training!')
        best_accuracy = 0.
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)
            if epoch != 1 and self.probabilities[epoch - 1] != 0:
                train, test = get_train_val_loaders_sstdata(self.probabilities, epoch - 1, self.sentiment_dict)
                self.train_loader = train

            self._train_epoch(epoch)
            test_acc = test_roberta(self.model, self.args.device, self.val_loader)

            if best_accuracy < test_acc:
                best_accuracy = test_acc
                print('New best accuracy. Model Saved!')
        return best_accuracy

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.args.device)
        pbar = tqdm(self.train_loader)
        correct = 0.
        processed = 0.
        step = 0
        length = len(self.train_loader)
        print(f"Length train loader {length}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.args.device)
            attention_masks = batch['attention_masks'].to(self.args.device)
            labels = batch['labels'].to(self.args.device)

            outputs = self.model(input_ids, attention_mask=attention_masks)
            y_pred = outputs.logits  
            loss = self.ce_loss(y_pred, labels)
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            processed += len(labels)
            step += 1
            pbar.set_description(desc=f'Loss={loss.item():0.3f} Accuracy={100 * correct / processed:0.2f}')

        print(f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        return 100 * correct / processed
