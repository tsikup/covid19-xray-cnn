# COVID-19 X-ray CNN Classifier

The CNN model of our paper "TITLE", for predicting COVID19 from X-ray chest images against a) Pneumonia cases, b) Normal and Pneumonia cases and c) Normal, Bacterial Pneumonia and Viral Pneumonia cases.

Cite:
REFERENCE

Link:
URL

## How to use
### Model Training/Testing
In order to train or test the model you have to run `python main.py -c configs/example_config_kfold.json`.
`configs/example_config.json` contains every configuration option that you need to adjust in order to either train or test the model.
Please see the example file for further information and adjust it properly for your needs.

### GradCAM
Apply gradcam by running `python main.py -c configs/example_config_gradcam.json`. Again adjust the configuration file for your needs.

### Results Analysis
The `results_analysis.py` python file saves a json file with every eprformance metric you need, as well as the prediction results and the ground truth.
Run it by `python results_analysis.py -c configs/example_config_metrics.json`.
They are not yet fully automated, and a lot of ad-hoc adjustments are needed to produce every results found in the paper. 
But it is in my TODO list to make it to produce every result automatically.
