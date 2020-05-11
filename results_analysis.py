from metrics.metrics import BinaryMetrics, MulticlassMetrics
from sklearn.preprocessing import LabelBinarizer
from utils.config import get_config_from_json
from pathlib import Path
import numpy as np
import os

config, _ = get_config_from_json('configs/config_metrics.json')
results, _ = get_config_from_json(config.results.file)

n_classes = len(config.dataset.classes)

prob_predictions = np.array(results.prob_predictions)
ground_truth = np.array(results.ground_truth)
categorical_ground_truth = ground_truth

n_samples = ground_truth.size

if( len(ground_truth.shape) == 1):
    # prob_predictions_1d = prob_predictions
    # prob_predictions = np.zeros((n_samples, ground_truth.max()+1), dtype=float)
    # prob_predictions[np.arange(n_samples), 0] = 1 - prob_predictions_1d[np.arange(n_samples)]
    # prob_predictions[np.arange(n_samples), 1] = prob_predictions_1d[np.arange(n_samples)]
    predictions = np.round(prob_predictions + 0.01).astype(int)
    categorical_ground_truth = np.zeros((n_samples, ground_truth.max()+1), dtype=int)
    categorical_ground_truth[np.arange(n_samples), ground_truth] = 1
elif( len(ground_truth.shape) == 2):
    predictions = np.argmax(prob_predictions, axis=-1)
    ground_truth = np.argmax(categorical_ground_truth, axis=-1)

if(n_classes == 2):
    metrics = BinaryMetrics(ground_truth, predictions, prob_predictions, config, categorical_ground_truth=categorical_ground_truth, include_raw_results=False)
else:
    metrics = MulticlassMetrics(ground_truth, predictions, prob_predictions, config, categorical_ground_truth=categorical_ground_truth, include_raw_results=False)

metrics.pprint()
metrics.save(mkdir=True)