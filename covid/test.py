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
           # class_mode='binary',
            shuffle=False)
        model = load_model(weights_path + 'model15.h5')
        model.layers.pop()
        predictions = model.predict_generator(test_generator, steps=len(test_generator), verbose=0)
        classes = list(np.argmax(predictions, axis=1))
        filenames = test_generator.filenames
        answer = []
        for i in range(len(filenames)):
            if ((predictions[i])[0] < 0.33):
                string = str(filenames[i]) + " - Normal"+" - "+str(predictions[i][0])
                answer.append(string)
            elif ((predictions[i])[0] >= 0.67):
                string = str(filenames[i]) + " - pneumonia" + " -  " +str(predictions[i][0])
 
                answer.append(string)
            else:
                string = str(filenames[i]) + " -Covid"  + " - "+str(predictions[i][0])

                answer.append(string)
        return answer
if __name__ == '__main__':
    predictions = TestCovidNet.test()
    print(predictions)
