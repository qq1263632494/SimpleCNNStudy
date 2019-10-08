from torch import nn


def make_vgg_block(in_channel, out_channel, convs, pool=True):
    net = []
    net.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1))
    net.append(nn.BatchNorm2d(out_channel))
    net.append(nn.ReLU(inplace=True))
    for i in range(convs - 1):
        net.append(nn.Conv2d(out_channel, out_channel, 3, 1))
        net.append(nn.BatchNorm2d(out_channel))
        net.append(nn.ReLU(inplace=True))
    if pool:
        net.append(nn.MaxPool2d(2))
    return nn.Sequential(*net)


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        net = []
        net.append(make_vgg_block(3, 64, 2))
        net.append(make_vgg_block(64, 128, 2))
        net.append(make_vgg_block(128, 256, 4))
        net.append(make_vgg_block(256, 512, 4))
        net.append(make_vgg_block(512, 512, 4, False))
        self.cnn = nn.Sequential(*net)
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
