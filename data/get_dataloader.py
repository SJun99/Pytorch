"""
이 폴더에 데이터셋 별로 스크립트 만들기
"""
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader

def get_dataloader(cfg):
    """
    cfg에 따라 Dataset 클래스 객체를 만들고 Dataloader 까지 만들어 리턴
    """
    dataset = None

    if cfg['type'] == 'Mnist':
        from data.mnist import Mnist
        dataset = Mnist(cfg)

    elif cfg['type'] == 'Cifar10':
        from data.cifar10 import Cifar10
        dataset = Cifar10(cfg)

    return DataLoader(dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True)


def test_dataloader(dataloader, cfg):
    """
    get_dataloader()로 dataloader 만들고 그걸로 for문을 돌리면서
    데이터가 제대로 나오는지 확인
    opencv로 이미지를 화면에 띄워서 확인할 것
    """
    classes = cfg['class']['Cifar10']
    reversed_classes = {v: k for k, v in classes.items()}

    for i, data in enumerate(dataloader):
        if i >= 1:
            break
        img, labels = data['img'][:4], data['label'][:4]
        img = torchvision.utils.make_grid(img)

        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        print(' '.join(f'{label.item():5d}' for label in labels))
        plt.show()

