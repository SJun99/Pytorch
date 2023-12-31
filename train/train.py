import cv2
import config as cfg
import torch
import numpy as np
from data.get_dataloader import get_dataloader
from model.get_model import get_model
from model.get_model import test_model


def train_main():
    dataloader = get_dataloader(cfg.DATA['type'], 'train', cfg.DATA['PATH'], cfg.BATCH_SIZE)

    train_data = next(iter(dataloader))
    train_features = train_data['img']
    train_labels = train_data['label']

    # 이미지와 정답(label)을 표시합니다.
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}")
    img = train_features
    img = np.array(img, dtype=np.uint8)
    label = train_labels
    cv2.imshow('img', img[1])
    print(f"Label: {label}")
    cv2.waitKey(0)

    train_features = train_features.permute(0, 3, 1, 2)
    test_output = test_model(cfg.MODEL, train_features)
    model = get_model(cfg.MODEL)
    loss = get_loss(cfg.LOSS)
    optimizer = get_optimizer(cfg.OPTIMIZER)

    trainer = Trainer(model, loss, optimizer, cfg.TRAIN)
    trainer.train(dataloader)

    dataloader = get_dataloader(cfg.DATA, 'validation')
    validater = Validater(model)
    validater.eval(dataloader)


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