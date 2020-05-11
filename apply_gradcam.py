# USAGE
# python apply_gradcam.py --image images/space_shuttle.jpg
# python apply_gradcam.py --image images/beagle.jpg
# python apply_gradcam.py --image images/soccer_ball.jpg --model resnet

# import the necessary packages
from gradcam.gradcam import GradCAM
from utils.config import process_config
from utils.utils import get_args
from models.covid import *
from data_loader.covid_data_loader import *
from pathlib import Path
import numpy as np
import argparse
import imutils
import glob
import cv2
import os

def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--config", type=str)
	ap.add_argument("-i", "--image", type=str)
	ap.add_argument("-o", "--output", type=str)
	args = vars(ap.parse_args())
	config = process_config(args['config'], dirs=False, config_copy=False)

	# Set number of gpu instances to be used
	# set_gpus(config)
	os.environ["CUDA_VISIBLE_DEVICES"] = config.devices.gpu.id

	# Creating Model
	print("[INFO] Creating model...")
	model_instance = COVID_Model(config)
	print("[INFO] Loading model's weights...")
	model_instance.load(config.tester.checkpoint_path)
	model = model_instance.model

	# Creating DataLoader instance, for preprocessing
	print("[INFO] Create DataLoader instance for preprocessing...")
	data_loader = COVIDDataLoader(config)

	imgpath = args['image']
	if os.path.isfile(imgpath):
		image_path = imgpath        
		apply_gradcam(image_path)
	elif os.path.isdir(imgpath):
		files = glob.glob(os.path.join(imgpath,'*','*'))
		for i, f in enumerate(files):
			print(f)
			apply_gradcam(f, args['output'], model, data_loader, config, i+1)

def apply_gradcam(image_path, output_path, model, data_loader, config, patientID):
	# Load and preprocess image
	orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
	resized = cv2.resize(orig, tuple(config.model.resize_shape))
	image = resized.astype(np.float64)
	image = np.expand_dims(image, axis=0)
	image = data_loader.train_datagen.standardize(image)

	ground_truth = os.path.basename(os.path.dirname(image_path))

	# use the network to make predictions on the input imag and find
	# the class label index with the largest corresponding probability
	preds = model.predict(image)
	i = np.argmax(preds[0])

	# initialize our gradient class activation map and build the heatmap
	cam = GradCAM(model, i)
	heatmap = cam.compute_heatmap(image)

	# resize the resulting heatmap to the original input image dimensions
	# and then overlay heatmap on top of the image
	heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
	(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

	## draw the predicted label on the output image
	#cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
	#cv2.putText(output, str(i), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	#	0.8, (255, 255, 255), 2)

	# display the original image and resulting heatmap and output image
	# to our screen
	output = np.hstack([orig, output])
	output = imutils.resize(output, height=orig.shape[0])
	outputDir = os.path.join(output_path, ground_truth)
	Path(outputDir).mkdir(parents=True, exist_ok=True)
	cv2.imwrite(os.path.join(outputDir, 'p' + str(patientID) + '_' + str(i) + '.png'), output)

if __name__ == '__main__':
	main()
