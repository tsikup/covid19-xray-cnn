'''
A quick and naive way to split into kfold training and testing sets
'''

from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from glob import glob
import numpy as np
import shutil
import os


datasetDir = '/home/tsiknakisn/projects/covid19-xray/data/data/dataset/all'
kfoldDir = '/home/tsiknakisn/projects/covid19-xray/data/data/dataset/kfold'

images_0 = glob(os.path.join(datasetDir, '0/*'))
images_1 = glob(os.path.join(datasetDir, '1/*'))
images_2 = glob(os.path.join(datasetDir, '2/*'))
images_3 = glob(os.path.join(datasetDir, '3/*'))
images = np.array(images_0 + images_1 + images_2 + images_3)

print("Images shape: {}".format(images.shape))

labels = np.concatenate((np.zeros(len(images_0)), np.ones(len(images_1)), 2 * np.ones(len(images_2)), 3 * np.ones(len(images_3))))

print("Labels shape: {}".format(labels.shape))

skf = StratifiedKFold(n_splits=5, shuffle=True)

idx = 0
for train_index, test_index in skf.split(images, labels):
    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    outputTrainDir = os.path.join(kfoldDir, str(idx), 'train')
    outputTestDir = os.path.join(kfoldDir, str(idx), 'test')

    Path(os.path.join(outputTrainDir, '0')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outputTrainDir, '1')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outputTrainDir, '2')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outputTrainDir, '3')).mkdir(parents=True, exist_ok=True)
    
    Path(os.path.join(outputTestDir, '0')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outputTestDir, '1')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outputTestDir, '2')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outputTestDir, '3')).mkdir(parents=True, exist_ok=True)

    for idy, image in enumerate(X_train):
        try:
            imageName = os.path.basename(image)
            filepath = os.path.join(outputTrainDir, str(int(y_train[idy])), imageName)
            shutil.copy2(image, filepath)
        except Exception as e:
            print(e)

    for idy, image in enumerate(X_test):
      try:
          imageName = os.path.basename(image)
          filepath = os.path.join(outputTestDir, str(int(y_test[idy])), imageName)
          shutil.copy2(image,filepath)
      except Exception as e:
          print(e)
    idx = idx + 1
