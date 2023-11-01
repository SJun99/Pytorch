from torch import nn


class MlpClassifier(nn.Module):
    def __init__(self, data_input_feature, data_output_feature):
        super().__init__()

        self.fc_layer1 = nn.Linear(in_features=data_input_feature, out_features=784)
        self.fc_layer2 = nn.Linear(in_features=784, out_features=data_output_feature)

        # self.softmax_layer = nn.Linear(in_features=784, out_features=26)
        # 50176

    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        # 정규화 과정 추가

        return x
