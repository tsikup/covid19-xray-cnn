from base.base_tester import BaseTester
from metrics.metrics import BinaryMetrics, MulticlassMetrics
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

import sys

class COVIDModelTester(BaseTester):
    def __init__(self, model, data, config):
        super(COVIDModelTester, self).__init__(model, data, config)
        
    def test(self, save_metrics = True, return_metrics = True):
        n_classes = len(self.config.dataset.classes)
        predictions = np.asarray([], dtype=int)
        ground_truth = np.asarray([], dtype=int)
        categorical_ground_truth = np.asarray([], dtype=int)
        prob_predictions = np.asarray([], dtype=float)
        for iteration, data in enumerate(self.data):
            if(iteration > self.config.dataset.batch.test_size):
                break
            print('Batch iteration: {}'.format(iteration))
            x, y = data
            pred = self.model.predict(x)
            predictions = np.append(predictions, np.argmax(pred, axis=-1))
            ground_truth = np.append(ground_truth, np.argmax(y, axis=-1))
            categorical_ground_truth = np.append(categorical_ground_truth, np.reshape(y, -1))
            prob_predictions = np.append(prob_predictions, np.reshape(pred, -1))
        prob_predictions = np.reshape(prob_predictions, [-1, n_classes])
        categorical_ground_truth = np.reshape(categorical_ground_truth, [-1, n_classes])
        # Calculate and save confusion matrix and other metrics
        if (len(self.config.dataset.classes) == 2):
            metrics = BinaryMetrics(ground_truth, predictions, prob_predictions, self.config, categorical_ground_truth=categorical_ground_truth) # Create object's instance
        else:
            metrics = MulticlassMetrics(ground_truth, predictions, prob_predictions, self.config, categorical_ground_truth=categorical_ground_truth)
        if (save_metrics):
            metrics.pprint() # Print metrics
            metrics.save() # Save extended metrics
        if (return_metrics):
            return metrics
