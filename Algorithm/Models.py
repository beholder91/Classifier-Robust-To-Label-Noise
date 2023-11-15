import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5_28(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.2):
        super(LeNet5_28, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 4)
        self.fc1 = nn.Linear(480, 240)
        self.fc2 = nn.Linear(240, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(dropout_prob)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 480)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    


class LeNet5_32(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.2):
        super(LeNet5_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)  # Input: (3, 32, 32)  Output: (6, 32, 32)
        self.conv2 = nn.Conv2d(6, 16, 5)            # Input: (6, 16, 16) Output: (16, 12, 12)
        self.conv3 = nn.Conv2d(16, 120, 5)          # Input: (16, 6, 6)  Output: (120, 2, 2)
        self.fc1 = nn.Linear(480, 84)               # Input dimension 120*2*2, Output 84
        self.fc2 = nn.Linear(84, num_classes)       # Input dimension 84, Output num_classes
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 480)  # Reshape to align with fc1 input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x



