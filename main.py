from data_loader.covid_data_loader import *
from models.covid import *
from trainers.covid_trainer import *
from testers.covid_tester import *
from utils.config import process_config
from utils.utils import get_args
from utils.gpus import set_gpus
from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
import os

def train(config):
    print('Create the data generator.')
    data_loader = COVIDDataLoader(config)
    train_data, val_data = data_loader.get_train_data()
    test_data = data_loader.get_test_data()

    print('Create the model.')
    model = COVID_Model(config)

    print('Create the trainer.')
    trainer = COVIDModelTrainer(model.model, (train_data, val_data), config)

    print('Start training the model.')
    trainer.train()
    
    print('Create the tester.')
    tester = COVIDModelTester(model.model, test_data, config)

    print('Test the model.')
    tester.test()
    
def evaluate(config):    
    print('Create the data generator.')
    data_loader = COVIDDataLoader(config)
    test_data = data_loader.get_test_data()

    print('Create the model.')
    model = COVID_Model(config)
    
    print('Loading checkpoint\'s weights')
    model.load(config.tester.checkpoint_path)
    
    print('Create the tester.')
    tester = COVIDModelTester(model.model, test_data, config)
    
    print('Test the model.')
    tester.test()

def train_kfold(config):
    print('Create the data generator.')
    data_loader = COVIDKFold(config)
    data = data_loader.get_data()

    X,y = data.next()

    datasetKFold = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=None)

    datasetKFold.get_n_splits(X,y)

    idx = 0
    metrics = ()
    for train_index, test_index in datasetKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        config.dataset.batch.train_size = len(X_train)
        config.dataset.batch.test_size = len(X_test)

        print('Create the model.')
        model = COVID_Model(config)

        print('Create the trainer.')
        trainer = COVIDModelKFoldTrainer(model.model, (X_train, y_train), config)

        print('Start training the model.')
        trainer.train()
        
        print('Create the tester.')
        tester = COVIDModelTester(model.model, (X_test, y_test), config)

        print('Test the model.')
        metrics[idx] = tester.test(save_metrics=False, return_metrics=True)
        idx = idx + 1
    
    idx = 0
    for metric in metrics:
        metric.save('results_{}.json'.format(idx))
        idx = idx + 1
        

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config, dirs=True, config_copy=True)
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)

    # Set number of gpu instances to be used
    # set_gpus(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.devices.gpu.id
    
    print('Physical GPU devices: {}'.format(len(tf.config.experimental.list_physical_devices('GPU'))))
    print('Logical GPU devices: {}'.format(len(tf.config.experimental.list_logical_devices('GPU'))))
    
    if(config.mode == "train"):
        train(config)
    elif(config.mode == "eval"):
        evaluate(config)
    elif(config.mode == "train_kfold"):
        train_kfold(config)

if __name__ == '__main__':
    main()
