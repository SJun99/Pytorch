import torch
import torch.nn as nn

class Myloss(nn.Module):
    def __init__(self, cfg):
        super(Myloss, self).__init__()

        self.loss_type = cfg['type']

        if self.loss_type == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif self.loss_type == 'MSELoss':
            self.loss_func = nn.MSELoss()

    def forward(self, predictions, targets):
        if self.loss_type == 'CrossEntropy':
            loss = self.loss_func(predictions, targets)
            return loss
        elif self.loss_type == 'MSELoss':
            loss = self.loss_func(predictions, targets)
            return loss

def get_loss(cfg):
    return Myloss(cfg)
