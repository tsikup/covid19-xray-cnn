from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class DRModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(DRModelTrainer, self).__init__(model, data, config)
        self.train_generator = data[0]
        self.validation_generator = data[1]
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()
        
        self.train_steps_per_epoch = self.config.dataset.batch.train_size
        self.val_steps_per_epoch = self.config.dataset.batch.val_size

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if hasattr(self.config.toDict(),"comet_api_key"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch = self.train_steps_per_epoch,
            epochs = self.config.trainer.num_epochs,
            verbose = self.config.trainer.verbose_training,
            callbacks = self.callbacks,
            validation_data = self.validation_generator,
            validation_steps = self.val_steps_per_epoch,
            validation_freq = 1,
            initial_epoch = 0
        )
        # self.loss.extend(history.history['loss'])
        # self.acc.extend(history.history['acc'])
        # self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_acc'])
