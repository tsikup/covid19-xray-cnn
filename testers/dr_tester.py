from base.base_tester import BaseTester
from metrics.metrics import Metrics
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

class COVIDModelTester(BaseTester):
    def __init__(self, model, data, config):
        super(COVIDModelTester, self).__init__(model, data, config)
        
    def test(self):
        predictions = np.asarray([], dtype=int)
        ground_truth = np.asarray([], dtype=int)
        for iteration, data in enumerate(self.data):
            if(iteration>self.config.dataset.batch.test_size):
                break
            print('Batch iteration: {}'.format(iteration))
            x, y = data
            predictions = np.append(predictions, np.argmax(self.model.predict(x), axis=-1))
            ground_truth = np.append(ground_truth, np.argmax(y, axis=-1))
        # Calculate and save confusion matrix and other metrics
        metrics = Metrics(ground_truth, predictions, self.config) # Create object's instance
        metrics.pprint() # Print metrics
        metrics.save() # Save extended metrics
