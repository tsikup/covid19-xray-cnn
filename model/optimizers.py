import tensorflow as tf
import warnings
from base.base_optimizer import BaseOptimizer

class Optimizer(BaseOptimizer):
    def __init__(self, config):
        super(Optimizer, self).__init__(config)
        self.set_lr_schedule()
        
    def set_lr_schedule(self):
        try:
            if(self.config.model.lr_schedule_strategy == 'exp'):
                self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                        initial_learning_rate=self.config.model.learning_rate,
                                        decay_steps=100000,
                                        decay_rate=0.96,
                                        staircase=True
                                    )
            elif(self.config.model.lr_schedule_strategy == 'poly'):
                self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                                        starter_learning_rate=self.config.model.learning_rate,
                                        decay_steps=100000,
                                        end_learning_rate=self.config.model.learning_rate/1e3,
                                        power=0.5
                                    )
            else:
                self.lr_schedule=None
        except:
            self.lr_schedule=None
        
    def get(self):
        try:
            self.optimizer = self.optimizers[self.optimizer_name](learning_rate=self.lr_schedule if self.lr_schedule else self.config.model.learning_rate)
        except KeyError:
            warnings.warn("KeyError, {} optimizer not found. Falling back to Adam (default)".format(self.optimizer_name))
            self.optimizer = self.optimizers['adam'](learning_rate=self.lr_schedule if self.lr_schedule else self.config.model.learning_rate)
        finally:
            return self.optimizer