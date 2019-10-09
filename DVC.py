import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from FinalMakeResNet import get_ResNet
from TrainingTools import Classifier

IMG_SIZE = 64
train_data = np.load('train_data.npy')
train, val = train_test_split(train_data, test_size=0.25)

X_train = np.array([i[0] for i in train]).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
Y_train = np.array([i[1] for i in train])

X_val = np.array([i[0] for i in val]).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
Y_val = np.array([i[1] for i in val])


class DogVSCat(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.FloatTensor(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


cfg = [{'channels': 32, 'num_blocks': 2, 'stride': 1},
       {'channels': 64, 'num_blocks': 2, 'stride': 2},
       {'channels': 128, 'num_blocks': 2, 'stride': 2},
       {'channels': 256, 'num_blocks': 2, 'stride': 2},
       {'channels': 512, 'num_blocks': 2, 'stride': 2}]


train_set1 = torchvision.datasets.MNIST(
    root='./MNIST',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_set1 = torchvision.datasets.MNIST(
    root='./MNIST',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
print(train_set1.__getitem__(1)[0].shape)
train_set_dvc = DogVSCat(X=X_train, y=Y_train)
test_set_dvc = DogVSCat(X=X_val, y=Y_val)
print(train_set_dvc.__getitem__(1)[0].shape)
classifier = Classifier(nn=get_ResNet(cfg=cfg, num_classes=2,
                                      x=IMG_SIZE, ipt_channel=1))
classifier.fit(train_set=train_set_dvc, batch_size=250, optim='adam',
               loss_func=torch.nn.CrossEntropyLoss(), epoch=10,
               lr=0.005)
#classifier.fit_with_LBFGS(train_set=train_set_dvc, batch_size=100, loss_func=torch.nn.CrossEntropyLoss(), epoch=1, lr=0.0001)
classifier.evaluate(test_set=test_set_dvc, batch_size_test=250)
