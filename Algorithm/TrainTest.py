from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch
import matplotlib.pyplot as plt

def accuracy_top1(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels.data)

def train(model, optimizer, criterion, train_loader, val_loader, epochs=10, device='cuda'):
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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_corrects += accuracy_top1(outputs, labels)
            _, preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')

        train_metrics['loss'].append(train_loss)
        train_metrics['acc'].append(train_acc.item())
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
                val_corrects += accuracy_top1(outputs, labels)
                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')

        val_metrics['loss'].append(val_loss)
        val_metrics['acc'].append(val_acc.item())
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



# Plotting training and validation metrics
def plot(train_metrics, val_metrics):
    # Create subplots: 3 plots in a single row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plotting Training and Validation Loss
    axes[0].plot(train_metrics['loss'], label='Train Loss', color='r')
    axes[0].plot(val_metrics['loss'], label='Val Loss', color='b')
    axes[0].set_title('Loss vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plotting Training and Validation Accuracy
    axes[1].plot(train_metrics['acc'], label='Train Accuracy', color='r')
    axes[1].plot(val_metrics['acc'], label='Val Accuracy', color='b')
    axes[1].set_title('Accuracy vs. Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # Plotting Training and Validation F1 Score
    axes[2].plot(train_metrics['f1'], label='Train F1 Score', color='r')
    axes[2].plot(val_metrics['f1'], label='Val F1 Score', color='b')
    axes[2].set_title('F1 Score vs. Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# Test the model
def test(model, criterion, test_loader, device='cuda'):
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
            test_corrects += accuracy_top1(outputs, labels)
            _, preds = torch.max(outputs, 1)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())
                
    test_loss /= len(test_loader)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
    test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')

    print(f'Test Loss: {test_loss:.4f}, '
          f'Test Accuracy: {test_acc:.4f}, '
          f'Test F1: {test_f1:.4f}, '
          f'Test Precision: {test_precision:.4f}, '
          f'Test Recall: {test_recall:.4f}')
    
    return test_loss, test_acc, test_f1, test_precision, test_recall


