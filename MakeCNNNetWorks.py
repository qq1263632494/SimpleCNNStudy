def cal_conv2d(x=0, padding=0, dialation=1, kernel_size=0, stride=1):
    import math
    return math.floor((x + 2 * padding - dialation * (kernel_size - 1) - 1) / stride + 1)


def cal_maxpool2d(x=0, padding=0, dialation=1, kernel_size=0, stride=1):
    stride = kernel_size
    import math
    return math.floor((x + 2 * padding - dialation * (kernel_size - 1) - 1) / stride + 1)


height = 32
from torch import nn


def add_layers(cfg, in_channels, x, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            x = cal_maxpool2d(x=x, kernel_size=2, stride=2)
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            x = cal_conv2d(x, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers), x, in_channels


cfg = {
    'SimpleVGG': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256]
}


class VGG(nn.Module):
    def __init__(self, cfg, in_channels, x, batch_norm, num_classes=10):
        super(VGG, self).__init__()
        layers, width, channels = add_layers(cfg=cfg, in_channels=in_channels, x=x, batch_norm=batch_norm)
        self.layers = layers
        self.classifier = nn.Sequential(
            nn.Linear(channels * width * width, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
