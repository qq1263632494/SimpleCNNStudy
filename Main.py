import torch
import torchvision

from FinalMakeResNet import get_ResNet
from TrainingTools import Classifier

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                         download=True,
                                         transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                        download=True,
                                        transform=torchvision.transforms.ToTensor())
train_set_cifar100 = torchvision.datasets.CIFAR100(root='./CIFAR100', train=True,
                                                   download=False,
                                                   transform=torchvision.transforms.ToTensor())
test_set_cifar100 = torchvision.datasets.CIFAR100(root='./CIFAR100', train=False,
                                                  download=False,
                                                  transform=torchvision.transforms.ToTensor())
cfg = [{'channels': 32, 'num_blocks': 2, 'stride': 1},
       {'channels': 64, 'num_blocks': 2, 'stride': 2},
       {'channels': 128, 'num_blocks': 2, 'stride': 2},
       {'channels': 256, 'num_blocks': 2, 'stride': 2},
       {'channels': 512, 'num_blocks': 2, 'stride': 2}]
classifier = Classifier(nn=get_ResNet(cfg=cfg, num_classes=100, x=32))
# classifier = Classifier(nn=get_ResNet18())
classifier.fit(train_set=train_set, batch_size=1000, optim='sgd',
               loss_func=torch.nn.CrossEntropyLoss(), epoch=10,
               lr=0.005)
# classifier.fit_with_LBFGS(train_set=train_set, batch_size=25, loss_func=torch.nn.CrossEntropyLoss(), epoch=1, lr=0.0001)
classifier.evaluate(test_set=test_set, batch_size_test=50)
