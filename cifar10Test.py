import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from NetWorks import CNN3, CNN4

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
BATCH_SIZE = 5000
BATCH_SIZE_Test = 10000

LR = 0.01
EPOCH = 10
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_Test, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def draw(obj):
    img = obj[0]
    label = obj[1]
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


print(len(train_set))
print(len(test_set))

cnn = CNN4()
cnn = cnn.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


def train_1():
    i = 0
    list_x = []
    list_y = []
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x_t = b_x.cuda()
        b_y_t = b_y.cuda()
        for t in range(EPOCH):
            output = cnn(b_x_t)
            loss = loss_func(output, b_y_t)
            list_x.append(i)
            list_y.append(loss.item())
            i += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(list_x[0])
    print(list_y[0])
    print(list_x[-1])
    print(list_y[-1])
    plt.plot(list_x, list_y)
    plt.show()


def train_lbfgs(LR, msg):
    optimizer = torch.optim.LBFGS(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    list_x = []
    list_y = []
    train_loader.__setattr__('shuffle', True)
    from progressbar import ShowProcess
    bar = ShowProcess(EPOCH * len(train_set) / BATCH_SIZE)
    for t in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            def closure():
                b_x_t = b_x.cuda()
                b_y_t = b_y.cuda()
                output = cnn(b_x_t)
                loss = loss_func(output, b_y_t)
                list_y.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(closure)
            bar.show_process()
    bar.close()
    for i in range(len(list_y)):
        list_x.append(i)
    print(list_x[0])
    print(list_y[0])
    print(list_x[-1])
    print(list_y[-1])
    plt.title(msg)
    plt.text(.64, 7, list_y[-1])
    plt.plot(list_x, list_y)
    plt.show()


def train_without_closure(optimizer, msg):
    i = 0
    list_x = []
    list_y = []
    train_loader.__setattr__('shuffle', True)
    from progressbar import ShowProcess
    bar = ShowProcess(EPOCH)
    for t in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x_t = b_x.cuda()
            b_y_t = b_y.cuda()
            output = cnn(b_x_t)
            loss = loss_func(output, b_y_t)
            list_x.append(i)
            list_y.append(loss.item())
            i += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bar.show_process()
    bar.close()
    print(list_x[0])
    print(list_y[0])
    print(list_x[-1])
    print(list_y[-1])
    plt.title(msg)
    plt.plot(list_x, list_y)
    plt.show()


# train_without_closure(torch.optim.ASGD(cnn.parameters(), lr=LR))
train_lbfgs(0.1, 'Use LBFGS AND LR = 0.1')
list_pred = []
list_true = []
for step, (t_x, t_y) in enumerate(test_loader):
    t_x_t = t_x.cuda()
    test_output = cnn(t_x_t)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
    list_pred.append(pred_y)
    list_true.append(t_y)
print(list_pred[0])
print(list_true[0])
from sklearn.metrics import accuracy_score

print(accuracy_score(y_true=list_true[0], y_pred=list_pred[0]))
# print(accuracy_score(y_true=list_true[1], y_pred=list_pred[1]))
