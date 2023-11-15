from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch
import numpy as np

def accuracy_top1(outputs, labels):
    _, preds = torch.max(outputs, 1)
    _, true_labels = torch.max(labels, 1)
    corrects = torch.sum(preds == true_labels)
    return corrects

def train_mixup(model, optimizer, criterion, train_loader, val_loader, epochs=10, device='cuda', alpha=1.0):
    train_metrics = {'loss': [], 'acc': [], 'f1': [], 'pre': [], 're': []}
    val_metrics = {'loss': [], 'acc': [], 'f1': [], 'pre': [], 're': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        all_train_labels = []
        all_train_preds = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Perform mixup
            lam = np.random.beta(alpha, alpha)
            batch_size = labels.size(0)
            index = torch.randperm(batch_size).to(device)
            
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]
            
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_corrects += accuracy_top1(outputs, mixed_labels).cpu().numpy()
            
            _, preds = torch.max(outputs, 1)
            _, true_labels = torch.max(mixed_labels, 1)
            all_train_labels.extend(true_labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = train_corrects / len(train_loader.dataset)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')

        train_metrics['loss'].append(train_loss)
        train_metrics['acc'].append(train_acc)
        train_metrics['f1'].append(train_f1)
        train_metrics['pre'].append(train_precision)
        train_metrics['re'].append(train_recall)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_val_labels = []
        all_val_preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_corrects += accuracy_top1(outputs, labels).cpu().numpy()
                
                _, preds = torch.max(outputs, 1)
                _, true_labels = torch.max(labels, 1)
                all_val_labels.extend(true_labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = val_corrects / len(val_loader.dataset)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')

        val_metrics['loss'].append(val_loss)
        val_metrics['acc'].append(val_acc)
        val_metrics['f1'].append(val_f1)
        val_metrics['pre'].append(val_precision)
        val_metrics['re'].append(val_recall)
        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs} - '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Accuracy: {train_acc:.4f}, '
                f'Train F1: {train_f1:.4f}, '
                f'Train Precision: {train_precision:.4f}, '
                f'Train Recall: {train_recall:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_acc:.4f}, '
                f'Val F1: {val_f1:.4f}, '
                f'Val Precision: {val_precision:.4f}, '
                f'Val Recall: {val_recall:.4f}'
                )

    return train_metrics, val_metrics


def test_mixup(model, criterion, test_loader, device='cuda'):
    test_metrics = {'loss': 0.0, 'acc': 0, 'f1': 0, 'pre': 0, 're': 0}

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_corrects += accuracy_top1(outputs, labels).cpu().numpy()

            _, preds = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            all_test_labels.extend(true_labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = test_corrects / len(test_loader.dataset)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
    test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')

    test_metrics['loss'] = test_loss
    test_metrics['acc'] = test_acc
    test_metrics['f1'] = test_f1
    test_metrics['pre'] = test_precision
    test_metrics['re'] = test_recall
    

    print(f'Test Loss: {test_loss:.4f}, '
          f'Test Accuracy: {test_acc:.4f}, '
          f'Test F1: {test_f1:.4f}, '
          f'Test Precision: {test_precision:.4f}, '
          f'Test Recall: {test_recall:.4f}')

    return test_metrics