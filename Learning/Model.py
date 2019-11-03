# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:36:05 2019

@author: Julien Fischer

For a detailed explanation of the source code, consult the corresponding jupyter notebook file
"""
import sys
import warnings
if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=FutureWarning)
import cv2
import numpy as np
import os
from keras.utils import Sequence # for data generator class
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Dropout, concatenate, Conv2DTranspose
from keras.layers import Add # for skip connections
from keras.utils import plot_model
import json # for saving training history
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import gmtime, strftime
import argparse


class DataGenerator(Sequence):
    '''Assumes that examples in the provided folder are named from 1 to n, with n being the number of images'''
    def __init__(self, path_to_data_set='data/train', batch_size=32, image_size=(480,640), shuffle=True, scale_images=False):
        self.path_to_data = path_to_data_set
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.scale_images = scale_images
        self.training_size = self.__get_training_data_size(self.path_to_data)
        self.on_epoch_end()
        
        
    def __get_training_data_size(self, path_to_data):
        '''gets the number of samples'''
        path_color = os.path.join(path_to_data,'Color')
        if os.path.isdir(path_color):
            size = len([color for color in os.listdir(path_color) if os.path.isfile(os.path.join(path_color, color))])
            return size
        else:
            return 0
        
        
    def __len__(self):
        '''Number of batches per epoche'''
        return int(np.floor(self.training_size / self.batch_size))
    
    
    def on_epoch_end(self):
        '''Update indices (and their ordering) after each epoch'''
        # image names start with 1, np.arange(n,m) returns values from n to (m-1)
        self.indices = np.arange(1, self.training_size+1)
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
            
    def __data_generation(self, list_images):
        '''Generates data of size batch_size''' # X = (batch_size, 480, 640, 1)
        if self.scale_images == False:
            X1 = np.empty((self.batch_size, *self.image_size, 3), dtype=np.uint8) # color images
            X2 = np.empty((self.batch_size, *self.image_size), dtype=np.uint16) # ir image
        else:
            X1 = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32) # color images
            X2 = np.empty((self.batch_size, *self.image_size), dtype=np.float32) # ir image
        y = np.empty((self.batch_size, *self.image_size), dtype=np.uint16)  # depth image
        # Generate data
        for idx, name in enumerate(list_images):
            # load images in arrays
            img = cv2.imread(os.path.join(self.path_to_data, 'Color', str(name)+".jpg"), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.scale_images == False:
                X1[idx,] = img.astype(np.uint8)
            else:
                X1[idx,] = (img/255.).astype(np.float32)
            img = cv2.imread(os.path.join(self.path_to_data, 'Infrared', str(name)+".png"), cv2.IMREAD_ANYDEPTH)
            if self.scale_images == False:
                X2[idx,] = img.astype(np.uint16)
            else:
                X2[idx,] = (img/65535.).astype(np.float32)
            img = cv2.imread(os.path.join(self.path_to_data, 'Depth', str(name)+".png"), cv2.IMREAD_ANYDEPTH)
            y[idx,] = img.astype(np.uint16)
        # reshape ir and depth images
        X2 = X2.reshape(self.batch_size, 480, 640, 1)
        y = y.reshape(self.batch_size, 480, 640, 1)  
        return X1, X2, y
    
    
    def __getitem__(self, index):
        '''Generate one batch of data, X1 contains 8-bit RGB images, X2 16-bit infrared images and y corresponding 16-bit depth images'''
        # Generate indices of data   
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X1, X2, y = self.__data_generation(indices)
        return [X1, X2], y
    
    
def Binary_Mean_Absolut_Error(y_true, y_pred):
    '''Binary mean absolut error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    loss = K.sum(
                K.sum(
                        K.abs(y_true - y_pred) * A_i, 
                        axis=(1,2,3)
                ) 
                    /
                K.sum(
                        A_i,
                        axis=(1,2,3)
                )
           )
    return loss
    

class VGG:
    '''Class that contains building blocks for a residual VGG-like autoencoder network'''
    def __init__(self):
        self.layer_counting = {}
        
        
    def Block(self, number_of_layers, units, kernel_size, padding, activation):
        '''A block of <number_of_layers> convolutions with batch normalization added AFTER the non-linearity'''
        def Input(z):
            for i in range(1,number_of_layers+1):
                name = 'Conv' + str(kernel_size[0]) + '-' + str(units)
                # make sure we have unique layer names
                if name in self.layer_counting:
                    self.layer_counting[name] += 1
                else:
                    self.layer_counting[name] = 1
                name += '_' + str(self.layer_counting[name])
                name_bn = name + '_BN'
                z = Conv2D(filters=units, kernel_size=kernel_size, padding=padding, activation=activation, name=name)(z)
                z = BatchNormalization(name=name_bn)(z)
            return z
        return Input
    
    
    def Residual_Downsampling_Block(self, units, kernel_size, padding, activation):
        '''A block with a strided convolution for downsampling an the start of a skip connection'''
        def Input(z):
            skip = z
            name = 'DownConv' + str(kernel_size[0]) + '-' + str(units)
            # make sure we have unique layer names
            if name in self.layer_counting:
                self.layer_counting[name] += 1
            else:
                self.layer_counting[name] = 1
            name += '_' + str(self.layer_counting[name])
            name_bn = name + '_BN'
            z = Conv2D(filters=units, kernel_size=kernel_size, strides=(2,2), padding=padding, activation=activation, name=name)(z)
            z = BatchNormalization(name=name_bn)(z)
            return z, skip
        return Input
    
    
    def Residual_Upsampling_Block(self, units, kernel_size, padding, activation):
        '''A block with a transposed convolution (also called deconvolution) and the incorporation of a provided skip connection'''
        def Input(z, skip):
            name = 'UpConv' + str(kernel_size[0]) + '-' + str(units)
            # make sure we have unique layer names
            if name in self.layer_counting:
                self.layer_counting[name] += 1
            else:
                self.layer_counting[name] = 1
            name += '_' + str(self.layer_counting[name])
            name_add = name + '_skip'
            name_bn = name + '_BN'
            z = Conv2DTranspose(filters=units, kernel_size=kernel_size, strides=(2,2), padding="same", name=name)(z)
            z = Add(name=name_add)([z, skip])
            z = Activation(activation)(z)
            z = BatchNormalization(name=name_bn)(z)
            return z
        return Input
    
    
    def Residual_Block(self, number_of_layers, units, kernel_size, padding, activation):
        '''A block of <number_of_layers> covolutions with provided skip connection incorporated after the last convolutional layer'''
        def Input(z, skip):
            for i in range(1, number_of_layers+1):
                name = 'Conv2D' + str(kernel_size[0]) + '-' + str(units)
                # make sure we have unique layer names
                if name in self.layer_counting:
                    self.layer_counting[name] += 1
                else:
                    self.layer_counting[name] = 1
                name += '_' + str(self.layer_counting[name])
                name_add = name + '_skip'
                name_bn = name + '_BN'
                z = Conv2D(filters=units, kernel_size=kernel_size, padding=padding)(z)
                if i == number_of_layers:
                    z = Add(name=name_add)([z, skip])
                z = Activation(activation)(z)
                z = BatchNormalization(name=name_bn)(z)
            return z
        return Input


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description="Training of a VGG-style autoencoder for depth map prediction")
    Parser.add_argument("-t", "--train", type=str, default=None, help="Path to folder that contains the training and validation examples")
    Parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size to train the network with")
    Parser.add_argument("-m", "--models", type=str, default="saved_models", help="Path to where to save the intermediate models to")
    Parser.add_argument("-l", "--logs", type=str, default="logs", help="Path to where to save the training history (tensorboard logs) to")
    Parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs to train the network on")
    Parser.add_argument("-n", "--name_model", type=str, default="model", help="Name under which the final model should be saved")
    Parser.add_argument("-s", "--shuffle", type=bool, default=True, help="Whether the batches be shuffled for each epoch or not")
    Parser.add_argument("-i", "--input_scale", type=bool, default=True, help="Whether input images should be scaled to the range of [0,1] or not")
    Parser.add_argument("-y", "--history", type=str, default="history", help="Name under which the final history should be saved")
    Parser.add_argument("-o", "--optimizer", type=str, default="adam", help="The optimizer to utilize for training. Can be either a standard keras optimizer, or one of the following: [None at the moment]")
    Parser.add_argument("-p", "--periods", type=int, default=1, help="Number of epochs after which to save the current model (and its weights). 1 means every epoche.")
    args = Parser.parse_args()
    
    if not args.train:
        print("No directory with training examples specified!")
        print("For help use --help")
        exit()
        
    if args.batch_size <= 0:
        print("Invalid batch size supplied!")
        exit()
        
    if args.epochs <= 0:
        print("Invalid number of epochs supplied!")
        exit()
    
    training_generator = DataGenerator(
            path_to_data_set=os.path.join(args.train, 'train'),
            batch_size=args.batch_size,
            image_size=(480,640),
            shuffle=args.shuffle,
            scale_images=args.input_scale
        )

    validation_generator = DataGenerator(
            path_to_data_set=os.path.join(args.train, 'validation'),
            batch_size=args.batch_size,
            image_size=(480,640),
            shuffle=args.shuffle,
            scale_images=args.input_scale
        )
    
    current_date = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    current_model_name = 'model_' + current_date + '_epoch_{epoch:04d}.h5'
    os.makedirs(args.models, exist_ok=True)
    checkpoint_path = os.path.join(args.models, current_model_name)
    current_log_name = 'log_' + current_date
    log_path = os.path.join(args.logs, current_log_name)

    checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=args.periods)

    tensorboard_callback = TensorBoard(
            log_dir=log_path,
            histogram_freq = 0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=True,
            update_freq="epoch")

    callback_list = [checkpoint_callback, tensorboard_callback]
    
    vgg = VGG()
    # Color branch
    input_color = Input(shape=(480,640,3), name="Color_Input")
    x = Model(inputs=input_color, outputs=input_color)
    
    # Infrared branch
    input_ir = Input(shape=(480,640,1), name="Infrared_Input")
    y = Model(inputs=input_ir, outputs=input_ir)

    # combine both branches
    combined = concatenate([x.output, y.output], name="Concatenate")

    # zeroth skip connection start --> to transfer original input images to the end of the network
    skip_zero = combined

    # VGG16 style encoder (configuration D)

    z = vgg.Block(number_of_layers=2, units=64, kernel_size=(3,3), padding="same", activation="relu")(combined)
    # max pooling replaced with strided convolution + first skip connection start
    z, skip_one = vgg.Residual_Downsampling_Block(units=64, kernel_size=(3,3), padding="same", activation="relu")(z)


    z = vgg.Block(number_of_layers=2, units=128, kernel_size=(3,3), padding="same", activation="relu")(z)
    # max pooling replaced with strided convolution + second skip connection start
    z, skip_two = vgg.Residual_Downsampling_Block(units=128, kernel_size=(3,3), padding="same", activation="relu")(z)


    z = vgg.Block(number_of_layers=3, units=256, kernel_size=(3,3), padding="same", activation="relu")(z)
    # max pooling replaced with strided convolution + third skip connection start
    z, skip_three = vgg.Residual_Downsampling_Block(units=256, kernel_size=(3,3), padding="same", activation="relu")(z)


    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu")(z)
    # max pooling replaced with strided convolution + fourth skip connection start
    z, skip_four = vgg.Residual_Downsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu")(z)


    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu")(z)
    # max pooling replaced with strided convolution + fifth skip connection start
    z, skip_five = vgg.Residual_Downsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu")(z)

    # end of encoder part

    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu")(z)

    # start of decoder part (= mirrored encoder part)

    # upsampling with deconvolution + fifth skip connection target
    z = vgg.Residual_Upsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu")(z, skip_five)
    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu")(z)

    # upsampling with deconvolution + fourth skip connection target
    z = vgg.Residual_Upsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu")(z, skip_four)
    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu")(z)


    # upsampling with deconvolution + third skip connection target
    z = vgg.Residual_Upsampling_Block(units=256, kernel_size=(3,3), padding="same", activation="relu")(z, skip_three)
    z = vgg.Block(number_of_layers=3, units=256, kernel_size=(3,3), padding="same", activation="relu")(z)


    # upsampling with deconvolution + second skip connection target
    z = vgg.Residual_Upsampling_Block(units=128, kernel_size=(3,3), padding="same", activation="relu")(z, skip_two)
    z = vgg.Block(number_of_layers=2, units=128, kernel_size=(3,3), padding="same", activation="relu")(z)


    # upsampling with deconvolution + first skip connection target
    z = vgg.Residual_Upsampling_Block(units=64, kernel_size=(3,3), padding="same", activation="relu")(z, skip_one)
    z = vgg.Block(number_of_layers=2, units=64, kernel_size=(3,3), padding="same", activation="relu")(z)

    # end of decoder part
    
    # TODO does incorporating skip_zero in this way makes sense?
    z = vgg.Residual_Block(number_of_layers=1, units=4, kernel_size=(3,3), padding="same", activation="relu")(z, skip_zero)

    # output layer
    z = Conv2D(1, kernel_size=(3,3), padding="same", name="Conv_Output")(z)

    model = Model(inputs=[x.input, y.input], outputs=z)

    model.compile(
            optimizer=args.optimizer,
            loss=Binary_Mean_Absolut_Error,
            metrics=['mae', 'mse', Binary_Mean_Absolut_Error])
    
    hist = model.fit_generator(
           generator=training_generator,
           validation_data=validation_generator,
           epochs=args.epochs,
           callbacks=callback_list)
    
    with open(args.history+'.json', 'w') as f:
        json.dump(hist.history, f)
    
    model.save(args.name_model+'.h5')
