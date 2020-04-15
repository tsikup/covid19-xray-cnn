'''
This code copies all covid-19 xray dicom images from portugal's dataset to be used as class 1 in the covid-19 deep learning classification model
'''

from glob import glob
import numpy as np
import shutil
import os

inputDir = '/home/tsik/Documents/github/covid19-xray/data/dx/' # repo folder
outputDir = '/home/tsik/Documents/github/covid19-xray/data/data/covid-19.portugal' # Output directory to store selected images

imageFolders = glob(os.path.join(inputDir,'covid-dx-*'))
imageFolders.sort()

# loop over the rows of the COVID-19 data frame
for idx in range(len(imageFolders)):
	try:
		inputPath = glob(os.path.join(imageFolders[idx], '*.dcm'))[0]
		filename = 'portugal-' + imageFolders[idx][-2:] + '.dcm'
		outputPath = os.path.join(outputDir, filename)
		if os.path.isfile(outputPath):
			continue
		shutil.copy2(inputPath, outputPath)
	except Exception as e:
		print(e)

