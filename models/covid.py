from base.base_model import BaseModel
from models.optimizers import Optimizer
from tensorflow.keras import Model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import *
import numpy as np

def COVID_Model(config):
    backbones = {
        "inception": COVID_InceptionV3,
        "resnet": COVID_ResNet50,
        "inception_resnet": COVID_InceptionResNetV2,
        "xception": COVID_Xception,
        "dense121": COVID_DenseNet121,
        "nasnet": COVID_NASNetLarge,
        "tsik": COVID_Tsik
    }
    classifiers = {
        "zhang": ZhangClassifier,
        "tsik": TsikClassifier
    }
    model = backbones[config.model.backbone](config, classifiers[config.model.classifier])
    return model

# ################# #
# Backbone networks #
# ################# #
class COVID_ResNet50(BaseModel):
    def __init__(self, config, classifier):
        super(COVID_ResNet50, self).__init__(config)
        self.classifier = classifier
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # ResNet50 as backbone network
        # Load pre-trained ResNet50 without the classifier
        self.backbone = ResNet50(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze ResNet50 parameters
        for layer in self.backbone.layers:
            layer.trainable = self.config.training.trainable
        
        self.output = self.classifier(self)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)

class COVID_InceptionV3(BaseModel):
    def __init__(self, config, classifier):
        super(COVID_InceptionV3, self).__init__(config)
        self.classifier = classifier
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # InceptionV3 as backbone network
        # Load pre-trained InceptionV3 without the classifier
        self.backbone = InceptionV3(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze InceptionV3 parameters
        for layer in self.backbone.layers:
            layer.trainable = self.config.training.trainable
        
        self.output = self.classifier(self)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)
    
class COVID_InceptionResNetV2(BaseModel):
    def __init__(self, config, classifier):
        super(COVID_InceptionResNetV2, self).__init__(config)
        self.classifier = classifier
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # InceptionResNetV2 as backbone network
        # Load pre-trained InceptionResNetV2 without the classifier
        self.backbone = InceptionResNetV2(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze InceptionResNetV2 parameters
        for layer in self.backbone.layers:
            layer.trainable = self.config.training.trainable
        
        self.output = self.classifier(self)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)
    
class COVID_Xception(BaseModel):
    def __init__(self, config, classifier):
        super(COVID_Xception, self).__init__(config)
        self.classifier = classifier
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # Xception as backbone network
        # Load pre-trained Xception without the classifier
        self.backbone = Xception(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze Xception parameters
        for layer in self.backbone.layers:
            layer.trainable = self.config.training.trainable
        
        self.output = self.classifier(self)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)
    

class COVID_DenseNet121(BaseModel):
    def __init__(self, config, classifier):
        super(COVID_DenseNet121, self).__init__(config)
        self.classifier = classifier
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # DenseNet121 as backbone network
        # Load pre-trained DenseNet121 without the classifier
        self.backbone = DenseNet121(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze DenseNet121 parameters
        for layer in self.backbone.layers:
            layer.trainable = self.config.training.trainable
        
        self.output = self.classifier(self)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)
    
class COVID_NASNetLarge(BaseModel):
    def __init__(self, config, classifier):
        super(COVID_NASNetLarge, self).__init__(config)
        self.classifier = classifier
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # NASNetLarge as backbone network
        # Load pre-trained NASNetLarge without the classifier
        self.backbone = NASNetLarge(include_top=False, input_tensor=self.visible, input_shape=self.input_shape, weights='imagenet')
        # (Un)Freeze NASNetLarge parameters
        for layer in self.backbone.layers:
            layer.trainable = self.config.training.trainable
        
        self.output = self.classifier(self)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])
        
    def predict(self, x):
        return self.model.predict(x)

class COVID_Tsik(BaseModel):
    def __init__(self, config, classifier):
        super(COVID_Tsik, self).__init__(config)
        self.classifier = classifier
        self.optimizer = Optimizer(config)
        self.build_model()

    def build_model(self):
        # Define input tensor
        self.visible = Input(shape=self.input_shape)

        # Conv Block 1
        self.conv1 = Conv2D(32, 3, 1, activation='relu', padding='same')(self.visible)
        self.pool1 = MaxPooling2D((2,2))(self.conv1)
        # Conv Block 2
        self.conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(self.pool1)
        self.pool2 = MaxPooling2D((2,2))(self.conv2)
        # Conv Block 3
        self.conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(self.pool2)
        self.pool3 = MaxPooling2D((2,2))(self.conv3)
        # Conv Block 4
        self.conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(self.pool3)

        self.output = self.classifier(self)

        # Define model
        self.model = Model(inputs=self.visible, outputs=self.output)
        
        self.model.summary()

        self.model.compile(
              loss = self.config.model.loss,
              optimizer = self.optimizer.get(),
              metrics = ["accuracy"])

    def get_feature_output(self):
        return self.conv4

    def predict(self, x):
        return self.model.predict(x)
    
# #################################### #
# Classifiers for pretrained backbones #
# #################################### #
def ZhangClassifier(model):
    ### Custom classifier
    ## W. Zhang, J. Zhong, S. Yang, Z. Gao, J. Hu, Y. Chen and Z. Yi, 
    ## "Automated identification and grading system of diabetic retinopathy using deep neural networks," 
    ## Knowledge-Based Systems, vol. 175, pp. 12-25, 1 7 2019.
    # GAP
    model.average_pooling = GlobalAveragePooling2D()(model.backbone.output)
    # Block 1
    model.hidden_1 = Dense(1024, activation='relu')(model.average_pooling)
    model.dropout_1 = Dropout(0.25)(model.hidden_1)
    # Block 2
    model.hidden_2 = Dense(512, activation='relu')(model.dropout_1)
    model.dropout_2 = Dropout(0.5)(model.hidden_2)
    # Block 3
    model.hidden_3 = Dense(256, activation='relu')(model.dropout_2)
    model.dropout_3 = Dropout(0.5)(model.hidden_3)
    # Block 4
    model.hidden_4 = Dense(128, activation='relu')(model.dropout_3)
    model.dropout_4 = Dropout(0.5)(model.hidden_4)
    # Output
    model.output = Dense(model.output_shape, activation='softmax')(model.dropout_4)
    return model.output

def TsikClassifier(model):
    # GAP
    model.average_pooling = GlobalAveragePooling2D()(model.get_feature_output())
    # Block 1
    model.hidden_1 = Dense(256, activation='relu')(model.average_pooling)
    model.dropout_1 = Dropout(0.25)(model.hidden_1)
    # Block 2
    model.hidden_2 = Dense(128, activation='relu')(model.dropout_1)
    model.dropout_2 = Dropout(0.25)(model.hidden_2)
    # Block 3
    model.hidden_3 = Dense(64, activation='relu')(model.dropout_2)
    model.dropout_3 = Dropout(0.25)(model.hidden_3)
    # Output
    model.output = Dense(model.output_shape, activation='softmax')(model.dropout_3)
    return model.output
