'''
A quick and naive way to split into training and testing sets
'''

from pathlib import Path
import numpy as np
import shutil
import os


datasetDir = '/home/tsik/Documents/github/covid19-xray/data/data/dataset/all'
trainDir = '/home/tsik/Documents/github/covid19-xray/data/data/dataset/train'
testDir  = '/home/tsik/Documents/github/covid19-xray/data/data/dataset/test'

classes = ['0', '1']

for c in classes:

    inputDir = os.path.join(datasetDir, c)
    outputTrainDir = os.path.join(trainDir, c)
    outputTestDir = os.path.join(testDir, c)

    Path(outputTrainDir).mkdir(parents=True, exist_ok=True)
    Path(outputTestDir).mkdir(parents=True, exist_ok=True)
    
    imageFiles = os.listdir(inputDir)
    indices = list(range(len(imageFiles)))
    np.random.shuffle(indices)
    numOfTestImages = round(len(imageFiles) * 0.2)

    indicesTrain = indices[:-numOfTestImages]
    indicesTest = indices[-numOfTestImages:]

    for idx in indicesTrain:
        try:
            filename = imageFiles[idx]
            inputPath = os.path.sep.join([inputDir,filename])
            outputPath = os.path.sep.join([outputTrainDir, filename])
            shutil.copy2(inputPath, outputPath)
        except Exception as e:
            print(e)
    
    for idx in indicesTest:
        try:
            filename = imageFiles[idx]
            inputPath = os.path.sep.join([inputDir,filename])
            outputPath = os.path.sep.join([outputTestDir, filename])
            shutil.copy2(inputPath, outputPath)
        except Exception as e:
            print(e)
