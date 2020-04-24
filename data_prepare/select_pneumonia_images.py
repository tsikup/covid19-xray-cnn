'''
This code selects a subset of x pneumonia xray images to be used as class 0 in the covid-19 deep learning classification model
'''

from pathlib import Path
import numpy as np
import shutil
import os

classes = ['normal', 'bacteria', 'virus']

for clas in classes:

	inputDir = os.path.join('/home/tsik/Documents/github/covid19-xray/data/pneumonia/images', clas) # repo folder
	outputDir = os.path.join('/home/tsik/Documents/github/covid19-xray/data/data/pneumonia_3class/', clas) # Output directory to store selected images

	Path(inputDir).mkdir(parents=True, exist_ok=True)
	Path(outputDir).mkdir(parents=True, exist_ok=True)

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

