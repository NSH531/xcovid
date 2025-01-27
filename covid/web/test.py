import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


class TestCovidNet:
    def __init__(self):
        self.model = None

    @staticmethod
    def imagenet_preproc(x):
        x = x / 255.0
        x = x - [0.485, 0.456, 0.406]
        x = x / [0.229, 0.224, 0.225]
        return x

    def test():
        test_data_path = "data/"
        input_size = 224
        batch_size = 1
        weights_path = "weights/"
        testcovidnet = TestCovidNet()
        test_datagen = ImageDataGenerator(horizontal_flip=True, preprocessing_function=testcovidnet.imagenet_preproc)
        test_generator = test_datagen.flow_from_directory(
            test_data_path,
            classes=['test'],
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        model = load_model(weights_path + 'model17.h5')
        model.layers.pop()
        predictions = model.predict_generator(test_generator, steps=len(test_generator), verbose=0)
        classes = list(np.argmax(predictions, axis=1))
        filenames = test_generator.filenames
        q='<header><nav>  <h1 height="33" style="background-color:red;color:white">    <span style="border-style: ridge ">xcovid</span>    <span style="border-style:dotted"><a href="/data" style="color:#AADDFF">data</a></span></h1>  </nav></header>  <section>'        
        q=q+'<div><br>'
        answer = []
        for i in range(len(filenames)):
            q=q+'  <div bgcolor="#FFFF22">'
            answer.append('<br>')
            q=q+str(filenames[i])
            if(predictions[i][2]>0.4):
                q=q+' <span COLOR="#ff1111">pneumonia</span></div>'
                #q=q+str(predictions[i][2])
            elif (predictions[i][1]>0.4):
                q=q+' - <pre color="#ff6600">covid19</pre>'
                #q=q+str(predictions[i][1])
            else:
                q=q+' - <pre color="#00ff44">normal</pre>'
#+str(predictions[i][0])
            #q=q+predictions[i]
        q=q+'</div>'
        return q
    