import tensorflow as tf

class BaseOptimizer(object):
    def __init__(self, config):
        self.config = config
        self.optimizer_name = self.config.model.optimizer
        self.init_learning_rate = self.config.model.learning_rate
        self.optimizers = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop
        }
        self.lr_schedule = None
        self.optimizer = None

    def set_lr_schedule(self):
        raise NotImplementedError
    
    def get(self, name, lr_schedule=None):
        raise NotImplementedError
