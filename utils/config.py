import json
from dotmap import DotMap
import os
import time
from shutil import copyfile
from utils.dirs import create_dirs

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file, dirs=True, config_copy=True):
    config, _ = get_config_from_json(json_file)
    if(config.exp.name):
        config.callbacks.tensorboard_log_dir = os.path.join("experiments", config.exp.name, "logs/", time.strftime("%Y-%m-%d-%H-%M/",time.localtime()))
        config.callbacks.checkpoint_dir = os.path.join("experiments", config.exp.name, "checkpoints/", time.strftime("%Y-%m-%d-%H-%M/",time.localtime()))
        if(config.mode == 'train' or config.mode =='train_kfold'):
            config.results.performance_dir = os.path.join("experiments", config.exp.name, "results/online", time.strftime("%Y-%m-%d-%H-%M/",time.localtime()))
            if dirs:
                print('Creating directories: {}, {}, {}'.format(config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.results.performance_dir))
                create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.results.performance_dir])
            if config_copy:
                copyfile(json_file, os.path.join(config.callbacks.checkpoint_dir,'config.json'))
        elif(config.mode == 'eval'):
            dirname = os.path.dirname(os.path.dirname(os.path.dirname(config.tester.checkpoint_path)))
            checkpoint = os.path.basename(config.tester.checkpoint_path)
            config.results.performance_dir = os.path.join(dirname,'results','offline',checkpoint, time.strftime("%Y-%m-%d-%H-%M/",time.localtime()))
            if dirs:
                print('Creating directories: {}'.format(config.results.performance_dir))
                create_dirs([config.results.performance_dir])
            if config_copy:
                copyfile(json_file, os.path.join(config.results.performance_dir,'config.json'))
    return config
