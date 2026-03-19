import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for binary classification/segmentation.
    Implements:
        FL = - w_pos * (1 - p)^gamma * log(p)
             - w_neg * p^gamma * log(1 - p)
    where p = sigmoid(x) is the probability of positive class.
    """
    def __init__(self, w_pos=1.0, w_neg=1.0, gamma=2.0, reduction='mean', eps=1e-8):
        """
        Args:
            w_pos (float): weight for positive class.
            w_neg (float): weight for negative class.
            gamma (float): focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
            eps (float): small epsilon to avoid log(0).
        """
        super().__init__()
        self.w_pos = w_pos
        self.w_neg = w_neg
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: raw model outputs (before sigmoid), shape (N, 1, H, W) or (N, H, W)
            targets: binary ground truth (0 or 1), same shape
        Returns:
            Scalar or tensor of loss.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        
        # Focal loss components
        pos_loss = -self.w_pos * (1 - probs) ** self.gamma * torch.log(probs) * targets
        neg_loss = -self.w_neg * (probs) ** self.gamma * torch.log(1 - probs) * (1 - targets)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":
    # Example for binary segmentation
    criterion = WeightedFocalLoss(w_pos=2.0, w_neg=1.0, gamma=2.0)
    
    logits = torch.randn(4, 1, 128, 128)  # model outputs
    targets = torch.randint(0, 2, (4, 1, 128, 128)).float() # values beetween 0 and 2

    print(targets.shape)
    
    loss = criterion(logits, targets)
    print("Weighted Focal Loss:", loss.item())
