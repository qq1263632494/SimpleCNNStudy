import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.

import cv2
import numpy as np  # dealing with arrays

from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion

TRAIN_DIR = './DogsVSCats/train'
IMG_SIZE = 64


def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return 0
    else:
        return 1


# 处理训练数据
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), label])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# train_data = create_train_data()