import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, img):
        x = self.backbone(img)
        x = self.head(x)
        return x

def get_model(cfg):
    """
    cfg에 따라 backbone, head를 조합하여 model 리턴
    """
    backbone = None
    head = None

    if cfg['backbone']['type'] == 'ResNet50':
        from model.backbone import ResNet50
        backbone = ResNet50(cfg['in_channels'])

    if cfg['head']['type'] == 'MlpClassifier':
        from model.head import MlpClassifier
        head = MlpClassifier(cfg['in_features'], cfg['out_features'])

    return MyModel(backbone, head)


def test_model(cfg, img):
    """
    get_model()로 model을 만들고 임의의 이미지를 넣었을 때 에러 없이 돌아가는지 확인
    backbone과 head의 호환성 확인
    head의 출력의 shape 확인
    이미지는 그냥 shape만 맞춘 torch.Tensor 만들어 사용
    """
    test_model = get_model(cfg)
    return test_model(img)



