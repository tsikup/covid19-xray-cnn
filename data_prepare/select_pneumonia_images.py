'''
This code selects a subset of x pneumonia xray images to be used as class 0 in the covid-19 deep learning classification model
'''

import numpy as np
import shutil
import os

inputDir = '/home/tsik/Documents/github/covid19-xray/data/pneumonia/images' # repo folder
outputDir = '/home/tsik/Documents/github/covid19-xray/data/data/pneumonia' # Output directory to store selected images

imageFiles = os.listdir(inputDir)
indices = list(range(len(imageFiles)))
np.random.shuffle(indices)
numOfImages = 150

# loop over the rows of the COVID-19 data frame
for idx in indices[:numOfImages]:
	try:
		filename = imageFiles[idx]
		inputPath = os.path.sep.join([inputDir,filename])
		outputPath = os.path.sep.join([outputDir, filename])
		shutil.copy2(inputPath, outputPath)
	except Exception as e:
		print(e)

