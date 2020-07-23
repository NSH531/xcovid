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
        test_data_path = "data/x/"
        input_size = 224
        batch_size = 1
        weights_path = "weights/x/"
        testcovidnet = TestCovidNet()
        test_datagen = ImageDataGenerator(horizontal_flip=True, preprocessing_function=testcovidnet.imagenet_preproc)
        test_generator = test_datagen.flow_from_directory(
            test_data_path,
            classes=['test'],
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        model = load_model( weights_path + 'modelx112.h5')
        model.layers.pop()
        predictions = model.predict_generator(test_generator, steps=len(test_generator), verbose=0)
        classes = list(np.argmax(predictions, axis=1))
        filenames = test_generator.filenames
        q='\n\n'
        right=0
        r=[]
        s1=0
        n=0
        nw=0
        si1=0
        wrong=0
        answer = []
        prediction = ''
        isPredictionCorrect = ''
        falsePositives = 0
        falseNegatives = 0   
        for i in range(len(filenames)):
            q=q+'\n'
            answer.append('\n')
            q=q+str(filenames[i])+'\n'
            q=q+'actual data:\t'+str(predictions[i])
            #t={(0,predictions[i][0]),(1,predictions[i][1]),(2,predictions[i][2])}
            #print (sorted(t))
            if(predictions[i][1]>predictions[i][0]):
                if( ("covid" in predictions[i]) or ("Covid" in predictions[i])or("COVID" in predictions[i])or("stage" in predictions[i])):
                    right=right+1
            else:
                if("Non" in predictions[i]):
                    right=right+1
                
        wrong = len(filenames)-right
        q=q+'\nTrue:\t'+str(right)
        q=q+'\nFalse:\t'+str(wrong)
        return q
if __name__ == '__main__':
    predictions = TestCovidNet.test()
    print(predictions)
