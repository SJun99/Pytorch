from torch import nn


class MlpClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_layer1 = nn.Linear(in_features=50176, out_features=784)
        self.fc_layer2 = nn.Linear(in_features=784, out_features=26)

    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)

        return x
