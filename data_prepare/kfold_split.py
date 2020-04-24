'''
A quick and naive way to split into kfold training and testing sets
'''

from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from glob import glob
import numpy as np
import shutil
import os


datasetDir = '../data/data/dataset/all'
kfoldDir = '../data/data/dataset/kfold'

images_0 = glob(os.path.join(datasetDir, '0/*'))
images_1 = glob(os.path.join(datasetDir, '1/*'))
images = np.array(images_0 + images_1)

labels = np.zeros(len(images_0))
labels = np.concatenate((labels, np.ones(len(images_1))))

skf = StratifiedKFold(n_splits=5, shuffle=True)

idx = 0
for train_index, test_index in skf.split(images, labels):
    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    outputTrainDir = os.path.join(kfoldDir, str(idx), 'train')
    outputTestDir = os.path.join(kfoldDir, str(idx), 'test')

    Path(outputTrainDir).mkdir(parents=True, exist_ok=True)
    Path(outputTestDir).mkdir(parents=True, exist_ok=True)

    for idy, image in enumerate(X_train):
        try:
            imageName = os.path.basename(image)
            filepath = os.path.join(outputTrainDir, str(int(y_train[idy])), imageName)
            shutil.copy2(image, filepath)
        except Exception as e:
            print(e)

    idx = idx + 1