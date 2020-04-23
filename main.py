from data_loader.covid_data_loader import *
from models.covid import *
from trainers.covid_trainer import *
from testers.covid_tester import *
from utils.config import process_config
from utils.utils import get_args
from utils.gpus import set_gpus
from sklearn.model_selection import StratifiedKFold

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

    idx = 0
    metrics = {}
    for train_gen, val_gen, test_gen in data_loader.split():
        print('Split: {}'.format(idx))
        print('Create the model.')
        model = COVID_Model(config)

        print('Create the trainer.')
        # trainer = COVIDModelKFoldTrainer(model.model, (X_train, y_train), config)
        trainer = COVIDModelTrainer(model.model, (train_gen, val_gen), config)

        print('Start training the model.')
        trainer.train()
        
        print('Create the tester.')
        # tester = COVIDModelTester(model.model, (X_test, y_test), config)
        tester = COVIDModelTester(model.model, test_gen, config)


        print('Test the model.')
        metrics[idx] = tester.test(save_metrics=False, return_metrics=True)
        metrics[idx].pprint()
        idx = idx + 1
    
    for idx, metric in enumerate(metrics):
        metric.save('results_{}.json'.format(idx))
        

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
