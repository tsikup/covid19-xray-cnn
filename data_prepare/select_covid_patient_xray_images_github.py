'''
This code finds all images of patients of a specified VIRUS and X-Ray view and stores selected image to an OUTPUT directory
+ It uses metadata.csv for searching and retrieving images name
+ Using ./images folder it selects the retrieved images and copies them in output folder
Code can be modified for any combination of selection of images
'''

import pandas as pd
import argparse
import shutil
import os

# Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
virus = "COVID-19" # Virus to look for
x_ray_view = "PA" # View of X-Ray

folder = '/home/tsik/Documents/github/covid19-xray/data/covid-chestxray-dataset/' # repo folder
metadata = os.path.join(folder, 'metadata.csv') # Meta info
imageDir = os.path.join(folder, 'images') # Directory of images
outputDir = '/home/tsik/Documents/github/covid19-xray/data/data/covid-19.pa' # Output directory to store selected images

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
	try:
		if row["finding"] != virus or row["view"] != x_ray_view:
			continue
		filename = row["filename"].split(os.path.sep)[-1]
		inputPath = os.path.sep.join([imageDir,filename])
		outputPath = os.path.sep.join([outputDir, filename])
		shutil.copy2(inputPath, outputPath)
	except Exception as e:
		print(e)

