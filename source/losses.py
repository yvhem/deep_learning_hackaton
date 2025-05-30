import torch

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, noise_prob=0.1):
        super().__init__()
        self.noise_prob = noise_prob
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        if self.training and torch.rand(1).item() < self.noise_prob:
            # Add label noise during training
            noisy_targets = targets.clone()
            num_classes = outputs.size(1)
            noise_mask = torch.rand(targets.size(), device=targets.device) < self.noise_prob
            noisy_targets[noise_mask] = torch.randint(0, num_classes, (noise_mask.sum(),), device=targets.device)
            return self.ce_loss(outputs, noisy_targets)
        else:
            return self.ce_loss(outputs, targets)