# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 09:17:59 2020

@author: ameyk
"""

import tensorflow.keras,os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import PIL.Image

trdata = ImageDataGenerator(horizontal_flip=True)
traindata = trdata.flow_from_directory(directory="train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="val", target_size=(224,224))

# example of tending the vgg16 model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(224, 224, 3))
model.summary()
for layer in model.layers:
    print(layer.output_shape)
print(model.output.shape)
# add new classifier layers
flat1 = Flatten()(model.output)
class1 = Dense(512, activation='relu')(flat1)
output = Dense(2, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()

from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=tensorflow.keras.losses.categorical_crossentropy, metrics=['accuracy'])

print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=1299//32,generator=traindata, validation_data= testdata, validation_steps=10,epochs=20,callbacks=[checkpoint,early])



model.save('vgg16_final.h5')