import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from test import test_bert, test_bert_20news
from transformers import AutoTokenizer
from dataloaders import get_train_val_loaders_sstdata
from sklearn.metrics import f1_score
#from newgroup import get_train_val_loaders_newsgroups
#from sklearn.datasets import fetch_20newsgroups
class TrainerBert:
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
        self.best_accuracy = 0. 
    def train(self):
        print('Training with Curriculum Learning!')
        best_accuracy = 0.

        # Define training stages
        stages = [(0.6, 1218), (0.8, 2436), (1.0, 12180)]  # (percentage of dataset, steps)

        for stage_idx, (fraction, max_steps) in enumerate(stages):
            print(f"\n=== Stage {stage_idx + 1}: Training on {fraction * 100:.0f}% of dataset for {max_steps} steps ===")


            # Get new dataloaders
            train, _ = get_train_val_loaders_sstdata(stage_idx+1)
            self.train_loader = train

            step = 0
            while step < max_steps:
                step = self._train_steps(step, max_steps)

        return best_accuracy

    
    def _train_steps(self, current_step, max_steps):
        """Train for a given number of steps."""
        self.model.train()
        self.model.to(self.args.device)
        pbar = tqdm(self.train_loader, total=max_steps, initial=current_step)
        correct, processed = 0, 0

        for batch in pbar:
            if current_step >= max_steps:
                break
            
            input_ids = batch['input_ids'].to(self.args.device)
            attention_masks = batch['attention_masks'].to(self.args.device)
            labels = batch['labels'].to(self.args.device)
            token_type_ids = batch['token_type_ids'].to(self.args.device)

            outputs = self.model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
            loss = self.ce_loss(outputs[0], labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = outputs[0].argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            processed += len(labels)

            current_step += 1
            pbar.set_description(f'Step {current_step}/{max_steps} - Loss={loss.item():.3f} Accuracy={100 * correct / processed:.2f}')
            
            if current_step % 1218 == 0 or current_step >= max_steps:
                test_acc = test_bert(self.model, self.args.device, self.val_loader)
                print(f"Validation Accuracy after {current_step} steps: {test_acc:.2f}")
                if test_acc > self.best_accuracy:
                    self.best_accuracy = test_acc
                    #self._save_model()
                    print(f'New best accuracy: {test_acc:.2f}. Model saved!')

        return current_step
    '''
    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.args.device)
        pbar = tqdm(self.train_loader)
        all_preds = []
        all_labels = []
        step = 0
        length = len(self.train_loader)
        print(f"Length train loader: {length}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.args.device)
            attention_masks = batch['attention_masks'].to(self.args.device)
            labels = batch['labels'].to(self.args.device)
            token_type_ids = batch['token_type_ids'].to(self.args.device)
            
            outputs = self.model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
            y_pred = outputs[0]
            loss = self.ce_loss(y_pred, labels)
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Store predictions and labels for F1 score calculation
            pred = y_pred.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(labels.cpu().numpy())
            
            # Optionally display intermediate F1 (less common)
            f1 = f1_score(all_labels, all_preds, average='macro')
            pbar.set_description(desc=f'Loss={loss.item():0.3f} F1={f1:0.4f}')
            
        # Final F1 score for the epoch
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Loss={loss.item()} F1 Score={f1:0.4f}')
        return f1
        '''