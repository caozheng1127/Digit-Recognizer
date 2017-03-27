# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "E:/PycharmProjects/files"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("E:/PycharmProjects/files/train.csv")

test_images = (pd.read_csv("E:/PycharmProjects/files/test.csv").values).astype('float32')

train_images = (train.ix[:,1:].values).astype('float32')
train_labels = train.ix[:,0].values.astype('int32')

result = train_images.shape
# print(result)


#Convert train datset to (num_images, img_rows, img_cols) format

train_images = train_images.reshape(train_images.shape[0],  28, 28)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
    plt.title(train_labels[i]);
    # plt.show()

train_images = train_images.reshape((42000, 28 * 28))

result = test_images.shape
# print(result)

num = train_labels.shape
# print(num)

num = train_labels
print(num)

