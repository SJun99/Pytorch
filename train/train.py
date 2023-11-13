import cv2
import config as cfg
import torch
import numpy as np
from data.get_dataloader import get_dataloader
from model.get_model import get_model
from model.get_model import test_model


def train_main():
    dataloader = get_dataloader(cfg.DATA, 'train')
    model = get_model(cfg.MODEL)
    loss = get_loss(cfg.LOSS)
    optimizer = get_optimizer(cfg.OPTIMIZER)

    trainer = Trainer(model, loss, optimizer, cfg.TRAIN)
    trainer.train(dataloader)

    dataloader = get_dataloader(cfg.DATA, 'validation')
    validater = Validater(model)
    validater.eval(dataloader)
# ResNet 직접 구현 해보기

def get_loss(cfg):
    if cfg['type'] == 'CrossEntropy':
        loss = torch.cross_entropy()

    return loss


def get_optimizer():
    pass


def main():
    train_main()

if __name__ == '__main__':
    main()