'''
This code converts dicom images to jpeg images
'''
from glob import glob
import numpy as np
import pydicom
import cv2
import os

inputDir  = '/home/tsik/Documents/github/covid19-xray/data/data/covid-19.portugal/dcm' # Input folder
outputDir = '/home/tsik/Documents/github/covid19-xray/data/data/covid-19.portugal/jpeg-2' # Output directory to store jpeg images

imageFilePaths = glob(os.path.join(inputDir,'*.dcm'))
imageFilePaths.sort()

# loop over the rows of the COVID-19 data frame
for imgFilePath in imageFilePaths:
    try:
        filename = os.path.basename(imgFilePath)
        # read dicom image
        ds = pydicom.dcmread(imgFilePath)
        # get image array
        img = ds.pixel_array
        # rescale/convert to 8-bit
        img = (img / img.max()) * 255.0
        # Convert to uint8
        img = np.uint8(img)
        # write jpeg image
        cv2.imwrite(os.path.join(outputDir, filename.replace('.dcm','.jpeg')),img)
    except Exception as e:
        print(e)

