from keras.models import Model
from keras.layers import (Input, concatenate, Conv2D, MaxPooling2D, 
    Conv2DTranspose, Dropout, BatchNormalization, UpSampling2D, Lambda)
import tensorflow as tf


class Unet:
    def __init__(self, outchns=1):
        self.outchns = outchns
        self.model = self.build()
        
    def loadWeights(self, pathToWeights):
        try:
            self.model.load_weights(pathToWeights)
        except:
            print(pathToWeights, 'cannot be loaded')
        
    def build(self):
        inputs = Input((None, None, 3))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Dropout(0.2)(conv5)
        
        up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        
        up7 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        
        up8 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
        
        up9 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)
        
        conv10 = Conv2D(self.outchns, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        return model
