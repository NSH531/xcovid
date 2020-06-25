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
        model = load_model(weights_path + 'modelCT11g12.h5')
        model.layers.pop()
        predictions = model.predict_generator(test_generator, steps=len(test_generator), verbose=0)
        classes = list(np.argmax(predictions, axis=1))
        filenames = test_generator.filenames
        q='\n\n'
        right=0
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
            if(predictions[i][0]>predictions[i][1]):
                prediction = 'normal'
                isPredictionCorrect = 'Non-' in filenames[i]
                
            else:
                prediction = 'covid'
                isPredictionCorrect = not('Non-' in filenames[i])
            if(isPredictionCorrect):
                right = right + 1
            else:
                if (prediction == 'normal'):
                    falseNegatives = falseNegatives + 1
                elif (prediction == 'covid'):
                    falsePositives = falsePositives + 1
                wrong = wrong + 1
            q=q+'\nPrediction:\t'+prediction+'\nCorrect:\t'+str(isPredictionCorrect)
        q=q+'\nTrue percent:\t'+str(100*right/(right+wrong))
        q=q+'\nFalse percent:\t'+str(100*wrong/(right+wrong))
        q=q+'\nFalse positives percent:\t'+str(100*falsePositives/wrong)
        q=q+'\nFalse negatives percent:\t'+str(100*falseNegatives/wrong)
        
        return q
if __name__ == '__main__':
    predictions = TestCovidNet.test()
    print(predictions)
