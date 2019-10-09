from torch import nn
from torch.nn import functional as F
from CNNUtils import cal_conv2d
from CNNUtils import cal_maxpool2d

INPUT_X = 0

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

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, x=32, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.INPUT_SIZE = x
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.INPUT_SIZE = cal_conv2d(x=self.INPUT_SIZE, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        print(self.INPUT_SIZE)
        self.fc = nn.Linear(512*self.INPUT_SIZE*self.INPUT_SIZE, num_classes)

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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_ResNet18():
    return ResNet(ResidualBlock)
