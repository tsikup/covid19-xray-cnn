# USAGE
# python apply_gradcam.py --image images/space_shuttle.jpg
# python apply_gradcam.py --image images/beagle.jpg
# python apply_gradcam.py --image images/soccer_ball.jpg --model resnet

# import the necessary packages
from gradcam import GradCAM
from utils.config import process_config
from utils.utils import get_args
from models.covid import *
from data_loader.covid_data_loader import *
import numpy as np
import argparse
import imutils
import cv2

# Set number of gpu instances to be used
# set_gpus(config)
os.environ["CUDA_VISIBLE_DEVICES"] = config.devices.gpu.id

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", type=str)
args = vars(ap.parse_args())

config = process_config(args.config, dirs=False, config_copy=False)

# Creating Model
print("[INFO] Creating model...")
model_instance = COVID_Model(config)
print("[INFO] Loading model's weights...")
model_instance.load(config.tester.checkpoint_path)
model = model_instance.model

# Creating DataLoader instance, for preprocessing
print("[INFO] Create DataLoader instance for preprocessing...")
data_loader = COVIDDataLoader(config)
# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions
print("[INFO] Load and preprocess image...")
orig = cv2.imread(config.image, cv2.IMREAD_COLOR)
resized = cv2.resize(orig, config.model.resize_shape)
image = resized.astype(np.float64)
image = np.expand_dims(image, axis=0)
image = data_loader.train_datagen.standardize(image)

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

# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, i, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.8, (255, 255, 255), 2)

# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)