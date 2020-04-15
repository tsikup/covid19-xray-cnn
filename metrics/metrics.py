from sklearn.metrics import confusion_matrix, roc_auc_score
from pprint import pprint
import json
import os
import numpy as np

class Metics():
    def __init__(self, ground_truth, predictions, config):
        self.config = config
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.con_matrix = confusion_matrix(ground_truth, predictions)
        self.tp = float(self.con_matrix[1][1])
        self.fp = float(self.con_matrix[1][0])
        self.tn = float(self.con_matrix[0][0])
        self.fn = float(self.con_matrix[0][1])
    
    def get_con_matrix(self):
        return self.con_matrix
    
    def get_sensitivity(self):
        return self.tp / (self.tp + self.fn)
    
    def get_specificity(self):
        return self.tn / (self.tn + self.fp)
    
    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fn + self.tn + self.fp)
    
    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_auc(self):
        return roc_auc_score(self.ground_truth, self.predictions)
    
    def get_tp(self):
        return self.tp
    
    def get_fp(self):
        return self.fp
    
    def get_tn(self):
        return self.tn
    
    def get_fn(self):
        return self.fn
    
    def json(self):
        results = {} # Iniate dictionary for storing several metrics
        results['confusion_matrix'] = {
            "tp": self.get_tp(),
            "fp": self.get_fp(),
            "tn": self.get_tn(),
            "fn": self.get_fn()
        }        
        results["spc"] = self.get_specificity()
        results["sen"] = self.get_sensitivity()
        results["pre"] = self.get_precision()
        results["acc"] = self.get_accuracy()
        results["auc"] = self.get_auc()
        return results
    
    def pprint(self):
        print('Results:\n')
        pprint(self.json())
        
    def save(self):
        np.save(os.path.join(self.config.results.performance_dir, 'confusion_matrix.npy'), self.get_con_matrix()) # Save confusion matrix
        with open(os.path.join(self.config.results.performance_dir, 'results.json'), "w") as fp:
            results_json = json.dumps(self.json())
            fp.write(results_json)
            fp.close()