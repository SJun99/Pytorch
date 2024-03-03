import cv2
import config as cfg
import torch
import numpy as np
from data.get_dataloader import get_dataloader, test_dataloader
from model.get_model import get_model
from model.get_model import test_model
from model.get_loss import get_loss
from model.get_optimizer import get_optimizer
from trainer import Trainer
from validater import Validater
from torch.utils.tensorboard import SummaryWriter


def train_main():
    dataloader = get_dataloader(cfg.DATA)
    test_dataloader(dataloader, cfg.DATA)
    model = get_model(cfg.MODEL)
    loss = get_loss(cfg.LOSS)
    optimizer = get_optimizer(cfg.OPTIMIZER, model.parameters())

    trainer = Trainer(dataloader, model, loss, optimizer, cfg.TRAIN)
    trainer.train()

    dataloader = get_dataloader(cfg.DATA)
    validater = Validater(model, cfg.DATA)
    validater.evaluation(dataloader)
    validater.individual_evaluation(dataloader)


def main():
    train_main()


if __name__ == '__main__':
    main()

