from base.base_data_loader import BaseDataLoader
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
import os

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
            color_mode=self.config.dataset.color_mode)

        self.validation_generator = self.train_datagen.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            subset = 'validation',
            color_mode=self.config.dataset.color_mode)
        
        self.test_generator = self.test_datagen.flow_from_directory(
            directory = self.config.dataset.test,
            target_size = self.config.model.resize_shape,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            color_mode=self.config.dataset.color_mode)

    def get_train_data(self):
        self.config.dataset.batch.train_size = len(self.train_generator)
        self.config.dataset.batch.val_size = len(self.validation_generator)
        return self.train_generator, self.validation_generator

    def get_test_data(self):
        self.config.dataset.batch.test_size = len(self.test_generator)
        return self.test_generator


class COVIDKFold(BaseDataLoader):
    def __init__(self, config):
        super(COVIDKFold, self).__init__(config)
        
        x, y = self._read_images()

        self.n_splits = self.config.dataset.n_splits

        datasetKFold = StratifiedKFold(n_splits=n_splits, shuffle=True)

        self.X_train = {}
        self.X_test = {}
        self.y_train = {}
        self.y_test = {}
        idx = 0
        for train_index, test_index in datasetKFold.split(x, y[:,1]):
            self.X_train[idx], self.X_test[idx] = x[train_index], x[test_index]
            self.y_train[idx], self.y_test[idx] = y[train_index], y[test_index]
        
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

    def split(self):
        for idx in n_splits:
            x_train = self.X_train[idx]
            y_train = self.y_train[idx]
            x_test = self.X_test[idx]
            y_test = self.y_test[idx]

            train_gen, val_gen = self.get_train_data(x_train, y_train)
            test_gen = self.get_test_data(x_test, y_test)

            yield train_gen, val_gen, test_gen


    def _read_images(self):
        generator = ImageDataGenerator()
        num_of_files = len(os.listdir(os.path.join(self.config.dataset.train, '0'))) + len(os.listdir(os.path.join(self.config.dataset.train, '1')))

        dataset = generator.flow_from_directory(
            directory = self.config.dataset.train,
            target_size = self.config.model.resize_shape,
            batch_size = num_of_files,
            class_mode = self.config.dataset.class_mode,
            shuffle = False,
            color_mode=self.config.dataset.color_mode)

        x, y = dataset.next()

        return x,y

    def get_train_data(self, X_train, y_train):
        self.train_generator = self.datagen.flow(
            x = X_train,
            y = y_train,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            subset = 'training')

        self.validation_generator = self.train_datagen.flow_from_directory(
            x = X_train,
            y = y_train,
            batch_size = self.config.trainer.batch_size,
            class_mode = self.config.dataset.class_mode,
            shuffle = True,
            subset = 'validation')

        self.config.dataset.batch.train_size = len(self.train_generator)
        self.config.dataset.batch.val_size = len(self.validation_generator)
        return self.train_generator, self.validation_generator

    def get_test_data(self, X_test, y_test):
        self.test_generator = self.test_datagen.flow_from_directory(
            x = X_test,
            y = y_test,
            batch_size = self.config.trainer.batch_size,
            shuffle = True)
        self.config.dataset.batch.test_size = len(self.test_generator)
        return self.test_generator
