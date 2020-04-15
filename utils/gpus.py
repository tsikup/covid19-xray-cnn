import os
import tensorflow as tf

def set_gpus(config):
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    gpus = config.devices.gpus
    if gpus and available_gpus:
        print('Working on GPUs/{}'.format(config.devices.gpus))
        gpus = [int(gpu) for gpu in gpus]
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.experimental.set_visible_devices(list(available_gpus[i] for i in gpus), 'GPU')
        except RuntimeError as e:
            print(e)
    else:
        print('Working on CPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""