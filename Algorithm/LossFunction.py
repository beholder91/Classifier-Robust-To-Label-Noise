import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseAwareCELoss(nn.Module):
    def __init__(self, Transition, device='cuda'):
        super(NoiseAwareCELoss, self).__init__()
        self.Transition = torch.tensor(Transition, dtype=torch.float32).to(device)
        self.base_criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # Adjust probabilities using the transition matrix
        probs = torch.softmax(logits, dim=1)
        adjusted_probs = torch.matmul(probs, self.Transition.t())

        # Calculate the adjusted loss
        one_hot_labels = F.one_hot(labels, num_classes=probs.shape[1]).float().to(probs.device)
        adjusted_loss = -torch.sum(torch.log(adjusted_probs + 1e-5) * one_hot_labels, dim=1)

        # Return the mean of adjusted loss
        return torch.mean(adjusted_loss)
