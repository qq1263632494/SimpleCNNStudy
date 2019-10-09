from torch import nn
from torch.nn import functional as F

from CNNUtils import cal_conv2d, cal_maxpool2d


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, x=0):
        super(ResidualBlock, self).__init__()
        self.x = x
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.x = cal_conv2d(x=self.x, kernel_size=3, stride=stride, padding=1)
        self.x = cal_conv2d(x=self.x, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            self.x = cal_conv2d(x=self.x, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_x(self):
        return self.x


LINEAR_OUTPUT = 2048


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, cfg, x=32, num_classes=10, ipt_channel=3):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.INPUT_SIZE = x
        self.conv1 = nn.Sequential(
            nn.Conv2d(ipt_channel, 16, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.INPUT_SIZE = cal_conv2d(x=self.INPUT_SIZE, kernel_size=2, stride=1, padding=1)
        layer = []
        self.cfg_length = len(cfg)
        for i in range(self.cfg_length):
            layer.append(self.make_layer(ResidualBlock, cfg[i]['channels'], cfg[i]['num_blocks'], cfg[i]['stride']))
            if i == self.cfg_length - 1:
                # print(self.INPUT_SIZE)
                self.INPUT_SIZE = cal_maxpool2d(x=self.INPUT_SIZE, kernel_size=2)
                # print('I am')
                # print(self.INPUT_SIZE)
                self.layers = nn.Sequential(*layer)
                self.pool = nn.MaxPool2d(kernel_size=2)
                self.fc = nn.Linear(LINEAR_OUTPUT, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.INPUT_SIZE = block(self.inchannel, channels, stride, x=self.INPUT_SIZE).get_x()
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_ResNet(cfg, x, num_classes, ipt_channel):
    return ResNet(cfg=cfg, x=x, num_classes=num_classes, ResidualBlock=ResidualBlock, ipt_channel=ipt_channel)
