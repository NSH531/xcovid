'''
Run with argument --weights with value 'imagenet' or 'last'
Examples:
python train.py --weights=imagenet
python train.py --weights=last
'''
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.applications import densenet
from keras.models import Model, load_model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import glob

class TrainCheXNet:
    def kullback_leibler_divergence(y_true, y_pred):

    def kullback_leibler_divergence(y_true,y_pred):
        from sklearn import svm,datasets
        from keras import backend as K
        from numpy import array
        import numpy as np
        q=0.0
        for i in y_pred:
            q=q+(y_true[i]*np.log(y_true[i]/y_pred[i])   
        return q

    def __init__(self):
        self.input_size = 224

        self.input = Input(shape=(self.input_size, self.input_size, 3))

        # Final dense layer will have single output since this is binary classification problem
        self.output_classes = 3

        # Following hyper-params are set as per the paper
        self.batch_size = 1
        self.decay_factor = 1.0/10.0


        self.val_batch_size = 1 # This can be set any convenient value as per GPU capacity

        #Following will be set by get_data_stats() based on the dataset
        self.w_class0 = None
        self.w_class1 = None
        self.train_steps = None
        self.val_steps = None
        self.model = None

    def get_data_stats(self, train_data_path, val_data_path, class_map):
        # Count images in each class
        cls_cnts = [0] * len(class_map)
        for key, value in class_map.items():
            imgs = glob.glob(train_data_path + value + "/*.jpeg")
            cls_cnts[key] = len(imgs)

        # compute class distribution
        self.w_class1 = float(cls_cnts[0])/sum(cls_cnts)
        self.w_class0 = float(cls_cnts[1])/sum(cls_cnts)

        # For convenience at train time, compute number of steps required to complete an epoch
        val_img_cnt = 0
        for key, value in class_map.items():
            imgs = glob.glob(val_data_path + value + "/*.jpeg")
            val_img_cnt += len(imgs)

        self.train_steps = (sum(cls_cnts) // self.batch_size) + 1
        self.val_steps = (val_img_cnt // self.val_batch_size) + 1

    def get_model(self):
        # DenseNet121 expects number of channels to be 3
        input = Input(shape=(self.input_size, self.input_size, 3))

        base_pretrained_model = densenet.DenseNet121(input_shape=(self.input_size, self.input_size, 3),
                                                     input_tensor=input, include_top=False, weights='imagenet')
        x = GlobalAveragePooling2D()(base_pretrained_model.layers[-1].output)
        x = Dense(self.output_classes, activation='sigmoid')(x)

        self.model = Model(inputs=input, outputs=x)

        # Note: default learning rate of 'adam' is 0.001 as required by the paper
        self.model.compile(optimizer='adam',  loss=TrainCheXNet.kullback_leibler_divergence(self.input,x))
        return self.model

    @staticmethod
    def imagenet_preproc(x):
        x = x / 255.0
        x = x - [0.485, 0.456, 0.406]
        x = x / [0.229, 0.224, 0.225]
        return x

    def train(self, train_data_path, val_data_path, epochs, weights_path):
        # We need to provide 'classes' to flow_from_directory() to make sure class 0 is 'normal'
        # and class 1 is covid
       # class_names = [0] * len(class_map)
       
# for key, value in class_map.items():
 #           class_names[key] = value

        # Paper suggests following:
        # 1. resize image to 224 x 224
        # 2. Use random horizontal flipping for augmenting
        # 3. normalize based on the mean and standard deviation of images in the ImageNet training set
        train_datagen = ImageDataGenerator(horizontal_flip=True,
                                           preprocessing_function=self.imagenet_preproc)
        train_generator = train_datagen.flow_from_directory(
            train_data_path,
           # classes=class_names,
            target_size=(self.input_size, self.input_size),
            batch_size=self.batch_size)
           # class_mode='binary')
        val_datagen = ImageDataGenerator(preprocessing_function=self.imagenet_preproc)
        val_generator = val_datagen.flow_from_directory(
            val_data_path,
           # classes=class_names,
            target_size=(self.input_size, self.input_size),
            batch_size=self.val_batch_size
           # class_mode='binary'
)

        # Paper suggests following:
        # 1. use an initial learning rate of 0.001 that is decayed by a factor of 10 each
        # time the validation loss plateaus after an epoch
        # 2. pick the model with the lowest validation loss

        checkpoint = ModelCheckpoint(weights_path + 'model16.h5', monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=self.decay_factor)

        callbacks = [checkpoint, reduceLROnPlat]

        model.fit_generator(generator=train_generator,
                            steps_per_epoch=self.train_steps,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_generator,
                            validation_steps=self.val_steps)
	
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True,
                                     metavar="'imagenet' or 'last'",
                                     help="Train on imagenet or last model.")
    args = parser.parse_args()
    train_data_path = "data/train/"
    val_data_path = "data/val/"
    class_map = {0:'normal', 1:'covid19',2:'pneumonia'}
    epochs =   13
    weights_path = "weights/"

    trainchexnet = TrainCheXNet()
    # Compute normal Vs Pneumonia class distribution
    trainchexnet.get_data_stats(train_data_path, val_data_path,class_map)

    # Create and compile the DenseNet121 model
    if (args.weights == 'imagenet'):
        model = trainchexnet.get_model()
    elif (args.weights == 'last'):
        model = load_model(weights_path + 'model16.h5')

    # Train the model
    trainchexnet.train(train_data_path, val_data_path, epochs, weights_path)
