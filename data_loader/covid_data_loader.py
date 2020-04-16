from base.base_data_loader import BaseDataLoader
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings

class COVIDDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(COVIDDataLoader, self).__init__(config)
        
        self.train_datagen = ImageDataGenerator(
            samplewise_center = True,
            samplewise_std_normalization = True,
            rotation_range = 15,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            validation_split = self.config.trainer.validation_split)
        
        self.test_datagen = ImageDataGenerator(
            samplewise_center = True,
            samplewise_std_normalization = True)

        self.train_generator = self.train_datagen.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            subset = 'training',
            color_mode="rgb")

        self.validation_generator = self.train_datagen.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            subset = 'validation',
            color_mode="rgb")
        
        self.test_generator = self.test_datagen.flow_from_directory(
            directory = self.config.dataset.test,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            color_mode="rgb")

    def get_train_data(self, classes=None):
            return self.train_generator, self.validation_generator

    def get_test_data(self, classes=None):
            return self.test_generator

