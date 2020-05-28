# Interpretable artificial intelligence framework for COVID‑19 screening on chest X‑rays

The CNN model of our paper "Interpretable artificial intelligence framework for COVID‑19 screening on chest X‑rays
", for predicting COVID19 from X-ray chest images against a) Pneumonia cases, b) Normal and Pneumonia cases and c) Normal, Bacterial Pneumonia and Viral Pneumonia cases.

Cite:
Tsiknakis, N., Trivizakis, E., Vassalou, E. E., Papadakis, G. Z., Spandidos, D. A., Tsatsakis, A., Sánchez‑García, J., López‑González, R., Papanikolaou, N., Karantanas, A. H., Marias, K."Interpretable artificial intelligence framework for COVID‑19 screening on chest X‑rays". Experimental and Therapeutic Medicine, https://doi.org/10.3892/etm.2020.8797

Link:
[https://www.spandidos-publications.com/10.3892/etm.2020.8797](https://www.spandidos-publications.com/10.3892/etm.2020.8797)

## How to use
### Model Training/Testing
In order to train or test the model you have to run `python main.py -c configs/config_kfold.json` for a kfold cross validation setting and `python main.py -c configs/config.json` for a setting with a single dataset.
`configs/config(_kfold).json` contains every configuration option that you need to adjust in order to either train or test the model. Note that after finishing the training, the model will be tested automatically. Use the eval option in the config to manually test a model without training it first.

The results will be saved under a created folder named, `experiments/name_of_the_setting/online(offline)/timestamp/`. If the testing is done manually (config's eval option), the results will be saved in the `offline` folder.

Please see the example file for further information and adjust it properly for your needs.

### GradCAM
Apply gradcam by running `python main.py -c configs/config_gradcam.json`. Again adjust the configuration file for your needs.

### Results Analysis
The `results_analysis.py` python file saves a json file with every performance metric you need, as well as the prediction results and the ground truth.
Run it by `python results_analysis.py -c configs/config_metrics.json`.

The analysis' results will be saved under a created folder named, `experiments/name_of_the_setting/analysis/`.
