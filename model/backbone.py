from torch import nn


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)

        return x
