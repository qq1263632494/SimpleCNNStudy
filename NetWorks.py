from torch import nn


class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Sequential(  # 3x32x32
            nn.Conv2d(
                in_channels=3,
                out_channels=10,
                kernel_size=4
            ),  # 10x29x29
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 10*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                kernel_size=3,
                out_channels=12
            ),  # 12*12*12
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)  # 12*6*6
        )
        self.out = nn.Linear(12 * 12 * 12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)\
        output = self.out(x)
        return output


class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        self.conv1 = nn.Sequential( #3*32*32
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=6
            ),  #16*27*27
            nn.MaxPool2d(kernel_size=3),    #16*9*9
            nn.ReLU()
        )
        self.output = nn.Linear(16 * 9 * 9, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
