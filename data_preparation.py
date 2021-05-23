import pandas as pd
import numpy as np
from PIL import Image

SIZE_IMAGE = 256
CARS = ['', '', '', '', '', '', '']


def label_to_vector(index):
    vector = np.zeros(len(CARS))
    vector[index] = 1.0
    return vector


data = pd.read_csv('anno_train.csv')
labels_train = []
images_train = []
index = 1
total = data.shape[0]

for index, row in data.iterrows():
    label = label_to_vector(row['label'])
    image = Image.open(row['image']).resize((256, 256))
    if image is not None:
            labels_train.append(label)
            images_train.append(image)
    else:
        print("Error")
    index += 1
    print("Progress train: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print("Total: " + str(len(images_train)))
np.save('./files/labels_train.npy', labels_train)
np.save('./files/images_train.npy', images_train)


data = pd.read_csv('anno_test.csv')
labels_test = []
images_test = []
index = 1
total = data.shape[0]

for index, row in data.iterrows():
    label = label_to_vector(row['label'])
    image = Image.open(row['image']).resize((256, 256))
    if image is not None:
            labels_test.append(label)
            images_test.append(image)
    else:
        print("Error")
    index += 1
    print("Progress test: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print("Total: " + str(len(images_test)))
np.save('./files/labels_test.npy', labels_test)
np.save('./files/images_test.npy', images_test)