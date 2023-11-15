import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Function to normalize the columns of a matrix
def normalize_columns(matrix):
    return matrix / matrix.sum(dim=0)

# Function to compute top-1 accuracy
def accuracy_top1(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels.data)

# Modified training function with device allocation and debugging info
def train_vmn(model, optimizer, optimizer_trans, criterion, T_hat, train_loader, val_loader, epochs=10, device='cuda', lambda_param=0.1):
    # Ensure T_hat is on the same device
    T_hat = T_hat.to(device)
    
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
            optimizer_trans.zero_grad()
            outputs = model(inputs)

            softmax_outputs = nn.Softmax(dim=1)(outputs)

            # Transform the outputs using T_hat
            # transformed_outputs = torch.mm(outputs, T_hat.t())
            transformed_outputs = torch.mm(softmax_outputs, T_hat)

            # Compute the cross-entropy loss
            # ce_loss = criterion(transformed_outputs, labels)
            ce_loss = criterion(torch.log(transformed_outputs + 1e-8), labels)

            # Compute the additional loss using logdet of T_hat
            _, logdet_loss = torch.slogdet(T_hat)
            logdet_loss = logdet_loss.abs()  # Take the absolute value of the log determinant

            # Check for invalid values
            if torch.isinf(logdet_loss) or torch.isnan(logdet_loss):
                print("Warning: Invalid logdet_loss encountered.")
                logdet_loss = torch.tensor(0.0).to(device)  

            # Combine the two losses
            total_loss = ce_loss + lambda_param * logdet_loss

            total_loss.backward()
            optimizer.step()
            optimizer_trans.step()

            # Normalize the columns of T_hat
            T_hat.data = normalize_columns(T_hat.data)

            train_loss += total_loss.item()
            train_corrects += accuracy_top1(transformed_outputs, labels)
            _, preds = torch.max(transformed_outputs, 1)
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
        
        # Log T_hat and logdet_loss for debugging
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} - T_hat: \n {T_hat.detach().cpu().numpy()}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_val_labels = []
        all_val_preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                softmax_outputs = nn.Softmax(dim=1)(outputs)
                transformed_outputs = torch.mm(softmax_outputs, T_hat)
                
                ce_loss = criterion(torch.log(transformed_outputs + 1e-8), labels)

                _, preds = torch.max(transformed_outputs, 1)
                val_loss += ce_loss.item()
                val_corrects += accuracy_top1(transformed_outputs, labels)
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




# Test the model with T_hat
def test_vmn(model, criterion, T_hat, test_loader, device='cuda'):
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    all_test_labels = []
    all_test_preds = []
    
    # Ensure T_hat is on the same device
    T_hat = T_hat.to(device)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            softmax_outputs = nn.Softmax(dim=1)(outputs)
            
            # Transform the outputs using T_hat
            transformed_outputs = torch.mm(softmax_outputs, T_hat)
            
            # Compute the cross-entropy loss
            ce_loss = criterion(torch.log(transformed_outputs), labels)
            
            test_loss += ce_loss.item()
            test_corrects += accuracy_top1(transformed_outputs, labels)
            _, preds = torch.max(transformed_outputs, 1)
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

