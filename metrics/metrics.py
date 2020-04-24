from sklearn.metrics import confusion_matrix, roc_auc_score
from pprint import pprint
import json
import os
import numpy as np

class Metrics():
    def __init__(self, ground_truth, predictions, prob_predictions, config, categorical_ground_truth=None):
        self.config = config
        self.ground_truth = ground_truth
        self.prob_predictions = prob_predictions
        self.con_matrix = confusion_matrix(ground_truth, predictions)
        self.categorical_ground_truth = categorical_ground_truth if categorical_ground_truth else ground_truth
        self.tp = float(self.con_matrix[1][1])
        self.fp = float(self.con_matrix[1][0])
        self.tn = float(self.con_matrix[0][0])
        self.fn = float(self.con_matrix[0][1])
    
    def get_con_matrix(self):
        return self.con_matrix
    
    def get_sensitivity(self):
        return self.tp / (self.tp + self.fn + 1e-8)
    
    def get_specificity(self):
        return self.tn / (self.tn + self.fp + 1e-8)
    
    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fn + self.tn + self.fp + 1e-8)
    
    def get_precision(self):
        return self.tp / (self.tp + self.fp + 1e-8)

    def get_auc(self):
        return roc_auc_score(self.categorical_ground_truth, self.prob_predictions)
    
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
        results["prob_predictions"] = self.prob_predictions.tolist()
        results["ground_truth"] = self.categorical_ground_truth.tolist()
        return results
    
    def pprint(self):
        print('Results:\n')
        pprint(self.json())
        
    def save(self, filename='results.json'):
        with open(os.path.join(self.config.results.performance_dir, filename), "w") as fp:
            results_json = json.dumps(self.json())
            fp.write(results_json)
            fp.close()
