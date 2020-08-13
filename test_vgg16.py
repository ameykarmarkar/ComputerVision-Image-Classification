# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:05:55 2020

@author: ameyk
"""



# example of using a pre-trained model as a classifier
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow
import os
import matplotlib.pyplot as plt
model = tensorflow.keras.models.load_model('vgg16_final.h5')

# Check its architecture
model.summary()

# load the model
#model = VGG16()
# load an image from file
count = 0
file = open('test_vc2kHdQ.csv','r')
out_file = open('submission.csv','w')
out_file.write('image_names,emergency_or_not\n')
for line in file:
    count += 1
    if count != 1:
        imageName = line.strip('\n')
        image = load_img(os.path.join('images', imageName), target_size=(224, 224))
        plt.imshow(image)
        
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        
        # predict the probability across all output classes
        yhat = model.predict(image)
        #print(yhat)
        
        if yhat[0][0] > yhat[0][1]:
            #print("emergency")
            out_file.write(imageName + ',1\n')
            plt.title("Emergency Vehicle")
        else:
            #print('non_emergency')
            out_file.write(imageName + ',0\n')
            plt.title("Non - Emergency Vehicle")
        plt.show()
file.close()
out_file.close()
# convert the probabilities to class labels
#label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
#label = label[0][0]
# print the classification
#Sprint('%s (%.2f%%)' % (label[1], label[2]*100))
