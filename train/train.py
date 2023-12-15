import cv2
import config as cfg
import torch
import numpy as np
from data.get_dataloader import get_dataloader
from model.get_model import get_model
from model.get_model import test_model
from model.get_loss import get_loss
from model.get_optimizer import get_optimizer
from trainer import Trainer
from validater import Validater
from torch.utils.tensorboard import SummaryWriter


def train_main():
    dataloader = get_dataloader(cfg.DATA)
    model = get_model(cfg.MODEL)
    loss = get_loss(cfg.LOSS)
    optimizer = get_optimizer(cfg.OPTIMIZER, model.parameters())

    # 텐서보드를 사용하여 손실값 로깅
    # writer = SummaryWriter()

    # for epoch in range(cfg.TRAIN['epochs']):
    #     # 훈련 데이터로더로부터 미니배치 가져오기
    #     for batch_idx, batch in enumerate(dataloader):
    #         # 입력 데이터와 레이블 추출
    #         inputs = batch['img']
    #         labels = batch['label']
    #
    #         # 모델 출력 계산
    #         outputs = model(inputs)
    #
    #         # 손실 계산
    #         loss_after = loss(outputs, labels)
    #
    #         # 옵티마이저 업데이트
    #         optimizer.zero_grad()
    #         loss_after.backward()
    #         optimizer.step()
    #
    #         # 각 미니배치에서 손실값 출력
    #         print(
    #             f'Epoch {epoch + 1}/{cfg.TRAIN["epochs"]}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss_after.item()}')

    trainer = Trainer(dataloader, model, loss, optimizer, cfg.TRAIN)
    trainer.train()

    dataloader = get_dataloader(cfg.DATA)
    validater = Validater(model)
    validater.eval(dataloader)


def main():
    train_main()


if __name__ == '__main__':
    main()
