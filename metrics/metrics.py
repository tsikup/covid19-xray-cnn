from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, roc_auc_score, classification_report
from pathlib import Path
from pprint import pprint
import json
import os
import numpy as np

class BinaryMetrics():
    def __init__(self, ground_truth, predictions, prob_predictions, config, categorical_ground_truth=None, include_raw_results=True):
        self.config = config
        self.include_raw_results = include_raw_results
        self.ground_truth = ground_truth
        self.categorical_ground_truth = categorical_ground_truth if categorical_ground_truth is not None else ground_truth
        self.prob_predictions = prob_predictions
        self.con_matrix = confusion_matrix(ground_truth, predictions)
        self.classification_report = classification_report(ground_truth, predictions, target_names=self.config.dataset.target_names, output_dict=True)
        self.analyze_con_matrix()

    def analyze_con_matrix(self):
        self.tn = float(self.con_matrix[0][0])
        self.fp = float(self.con_matrix[0][1])
        self.fn = float(self.con_matrix[1][0])
        self.tp = float(self.con_matrix[1][1])
    
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
        return roc_auc_score(self.ground_truth, self.prob_predictions)

    def get_classification_report(self):
        return self.classification_report
    
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
        results['confusion_matrix'] = self.get_con_matrix().tolist()
        results["spc"] = self.get_specificity()
        results["sen"] = self.get_sensitivity()
        results["pre"] = self.get_precision()
        results["acc"] = self.get_accuracy()
        results["auc"] = self.get_auc()
        if(self.include_raw_results == True):
            results["prob_predictions"] = self.prob_predictions.tolist()
            results["ground_truth"] = self.categorical_ground_truth.tolist()
        return results
    
    def pprint(self):
        print('Results:\n')
        pprint(self.json())
        pprint(self.classification_report)
        
    def save(self, filename='results.json', mkdir=False):
        if (mkdir == True):
            Path(self.config.results.performance_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.config.results.performance_dir, filename), "w") as fp:
            
            results_json = json.dumps(self.json())
            fp.write(results_json)
            fp.close()
        with open(os.path.join(self.config.results.performance_dir, 'classification_report.txt'), "w") as fp:
            report_json = json.dumps(self.classification_report)
            fp.write(report_json)
            fp.close()

class MulticlassMetrics():
    def __init__(self, ground_truth, predictions, prob_predictions, config, categorical_ground_truth=None, include_raw_results=True):
        self.config = config
        self.include_raw_results = include_raw_results
        self.n_classes = len(self.config.dataset.classes)
        self.ground_truth = ground_truth
        self.prob_predictions = prob_predictions
        self.basic_con_matrix = confusion_matrix(ground_truth, predictions)
        self.con_matrix = multilabel_confusion_matrix(ground_truth, predictions, labels=self.config.dataset.classes)
        self.categorical_ground_truth = categorical_ground_truth if categorical_ground_truth is not None else ground_truth
        self.classification_report = classification_report(ground_truth, predictions, target_names=self.config.dataset.target_names, output_dict=True)
        self.analyze_con_matrix()

    def analyze_con_matrix(self):
        self.basic_metrics = {}
        self.sensitivity = {}
        self.specificity = {}
        self.precision = {}
        self.accuracy = {}
        for i in range(self.n_classes):
            self.basic_metrics[i] = {
                'tn' : float(self.con_matrix[i][0][0]),
                'fp' : float(self.con_matrix[i][0][1]),
                'fn' : float(self.con_matrix[i][1][0]),
                'tp' : float(self.con_matrix[i][1][1])
            }
            tp = self.basic_metrics[i]['tp']
            fn = self.basic_metrics[i]['fn']
            tn = self.basic_metrics[i]['tn']
            fp = self.basic_metrics[i]['fp']
            self.specificity[i] = tn / (tn + fp + 1e-8)
            self.sensitivity[i] = tp / (tp + fn + 1e-8)
            self.precision[i] = tp / (tp + fp + 1e-8)
            self.accuracy[i] = (tp + tn) / (tp + tn + fn + fp + 1e-8)

    def get_basic_metrics(self):
        return self.basic_metrics
    
    def get_con_matrix(self):
        return self.basic_con_matrix
    
    def get_sensitivity(self):
        return self.sensitivity
    
    def get_specificity(self):
        return self.specificity
    
    def get_precision(self):
        return self.precision

    def get_accuracy(self):
        return self.accuracy

    def get_auc(self):
        return roc_auc_score(self.categorical_ground_truth, self.prob_predictions, multi_class='ovr'), roc_auc_score(self.categorical_ground_truth, self.prob_predictions, multi_class='ovo') 

    def get_classification_report(self):
        return self.classification_report
    
    def json(self):
        results = {} # Iniate dictionary for storing several metrics
        results["confusion_matrix"] = self.basic_con_matrix.tolist()
        results["spc"] = self.get_specificity()
        results["sen"] = self.get_sensitivity()
        results["pre"] = self.get_precision()
        results["acc"] = self.get_accuracy()
        results["auc_ovr"], results["auc_ovo"] = self.get_auc()
        if(self.include_raw_results == True):
            results["prob_predictions"] = self.prob_predictions.tolist()
            results["ground_truth"] = self.categorical_ground_truth.tolist()
        return results
    
    def pprint(self):
        print('Results:\n')
        pprint(self.json())
        pprint(self.classification_report)
        
    def save(self, filename='results.json', mkdir=False):
        if (mkdir == True):
            Path(self.config.results.performance_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.config.results.performance_dir, filename), "w") as fp:
            results_json = json.dumps(self.json())
            fp.write(results_json)
            fp.close()
        with open(os.path.join(self.config.results.performance_dir, 'classification_report.txt'), "w") as fp:
            report_json = json.dumps(self.classification_report)
            fp.write(report_json)
            fp.close()
