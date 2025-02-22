import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import f1_score
def test(model, device, test_loader):
    model.eval()
    final_loss = 0.
    correct = 0
    batch_num = 0.
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            final_loss += loss.item()
            batch_num += 1
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(final_loss / batch_num, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

def test_roberta(model, device, test_loader):
    model.eval()
    final_loss = 0.
    correct = 0
    batch_num = 0.

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits  
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            final_loss += loss.item()
            batch_num += 1

    avg_loss = final_loss / batch_num
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy
def test_bert(model, device, test_loader):
    model.eval()
    final_loss = 0.
    correct = 0
    batch_num = 0.
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids,  attention_mask=attention_masks, token_type_ids=token_type_ids)
            output = outputs[0]
            loss = F.cross_entropy(output, labels)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            final_loss += loss.item()
            batch_num += 1
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(final_loss / batch_num, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

def test_bert_20news(model, device, test_loader):
    model.eval()
    final_loss = 0.
    all_preds = []
    all_labels = []
    batch_num = 0.
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
            output = outputs[0]
            loss = F.cross_entropy(output, labels)
            
            pred = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(labels.cpu().numpy())
            
            final_loss += loss.item()
            batch_num += 1
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    print('\nTest set: Average loss: {:.4f}, F1 Score: {:.4f}\n'.format(final_loss / batch_num, f1))
    
    return f1