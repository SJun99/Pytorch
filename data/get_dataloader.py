"""
이 폴더에 데이터셋 별로 스크립트 만들기
"""
from torch.utils.data import DataLoader


def get_dataloader(cfg):
    """
    cfg에 따라 Dataset 클래스 객체를 만들고 Dataloader 까지 만들어 리턴
    """
    dataset = None

    if cfg == 'Mnist':
        from data.mnist import Mnist
        dataset = Mnist(**cfg['dataset'])

    elif cfg == 'Cifar10':
        from data.cifar10 import Cifar10
        dataset = Cifar10(mode, path)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_dataloader(cfg, mode, path, batch_size):
    """
    get_dataloader()로 dataloader 만들고 그걸로 for문을 돌리면서
    데이터가 제대로 나오는지 확인
    opencv로 이미지를 화면에 띄워서 확인할 것
    """
    dataset = None

    if cfg == 'Mnist':
        from mnist import Mnist
        dataset = Mnist(mode, path)

    elif cfg == 'Cifar10':
        from cifar10 import Cifar10
        dataset = Cifar10(mode, path)
        # 하나씩 띄우면서 확인 되도록 할 것

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

