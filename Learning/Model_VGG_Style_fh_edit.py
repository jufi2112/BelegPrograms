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
from keras.layers import Input, Conv2D, Activation, BatchNormalization, concatenate, Conv2DTranspose
from keras.layers import Add # for skip connections
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from time import gmtime, strftime
import argparse
import matplotlib.pyplot as plt
import pickle


class LearningRateDecay:
    '''Custom class to reduce learning rate after a specified event (for example after x epochs). 
    Code is taken from https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/'''
    def plot(self, epochs, title="Learning Rate Schedule"):
        lrs = [self(i) for i in epochs]
        
        N = np.arange(1,len(epochs)+1)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, lrs)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        
        
class StepDecay(LearningRateDecay):
    '''Custom class that implements step decay (i.e. drop learning rate after every x epochs)
    Code is taken from https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/'''
    def __init__(self, initAlpha=0.01, factor=0.5, dropEvery=10):
        self.initAlpha=initAlpha
        self.factor=factor
        self.dropEvery=dropEvery
    
    
    def __call__(self, epoch):
        exp = np.floor((1+epoch) / self.dropEvery)
        alpha = self.initAlpha * np.power(self.factor, exp)
        return float(alpha)





class DataGenerator(Sequence):
    '''Assumes that examples in the provided folder are named from 1 to n, with n being the number of images'''
    def __init__(self, path_to_data_set='data/train', batch_size=32, image_size=(480,640), shuffle=True, scale_images=False, scale_ground_truth=False):
        self.path_to_data = path_to_data_set
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.scale_images = scale_images
        self.scale_ground_truth = scale_ground_truth
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
        '''Generates data of size <batch_size>''' # X = (batch_size, 480, 640, 1)
        if self.scale_images == False:
            X1 = np.empty((self.batch_size, *self.image_size, 3), dtype=np.uint8) # color images
            X2 = np.empty((self.batch_size, *self.image_size), dtype=np.uint16) # ir image
        else:
            X1 = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32) # color images
            X2 = np.empty((self.batch_size, *self.image_size), dtype=np.float32) # ir image
        if self.scale_ground_truth:
            y = np.empty((self.batch_size, *self.image_size), dtype=np.float32)
        else:
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
            if self.scale_ground_truth:
                img = (img/65535.).astype(np.float32)
            y[idx,] = img
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
    
    
def WRONG_Masked_Mean_Absolute_Error(y_true, y_pred):
    '''Wrong version of masked mean absolut error custom loss function'''
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


def Masked_Mean_Absolute_Error(y_true, y_pred):
    '''Masked mean absolut error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    loss = K.mean(
                K.sum(
                        K.abs(y_true - y_pred) * A_i,
                        axis=(1,2,3)
                     )
                /
                K.sum(A_i, axis=(1,2,3))
            )
    lower_boundary = K.less(y_pred, 0)
    lower_boundary = K.cast(lower_boundary, dtype='float32')
    upper_boundary = K.greater(y_pred, 65535)
    upper_boundary = K.cast(upper_boundary, dtype='float32')
    interval_loss = K.sum(lower_boundary * 10000 + upper_boundary * 10000)   
    return loss+interval_loss


def Masked_Mean_Absolute_Error_Sigmoid(y_true, y_pred):
    '''Masked mean absolut error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    # Since we are using a sigmoid activation function, scale the predictions from [0,1] to [0,65535]
    y_pred = y_pred * 65535
    loss = K.mean(
                K.sum(
                        K.abs(y_true - y_pred) * A_i,
                        axis=(1,2,3)
                     )
                /
                K.sum(A_i, axis=(1,2,3))
            )
    lower_boundary = K.less(y_pred, 0)
    lower_boundary = K.cast(lower_boundary, dtype='float32')
    upper_boundary = K.greater(y_pred, 65535)
    upper_boundary = K.cast(upper_boundary, dtype='float32')
    interval_loss = K.sum(lower_boundary * 10000 + upper_boundary * 10000)   
    return loss+interval_loss


def Masked_Mean_Absolute_Error_Simple(y_true, y_pred):
    '''Masked mean absolut error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    loss = K.mean(
                K.sum(
                        K.abs(y_true - y_pred) * A_i,
                        axis=(1,2,3)
                     )
                /
                K.sum(A_i, axis=(1,2,3))
            ) 
    return loss


def Masked_Mean_Absolute_Error_Simple_Sigmoid(y_true, y_pred):
    '''Masked mean absolut error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    # Since we are using a sigmoid activation function, scale the predictions from [0,1] to [0,65535]
    y_pred = y_pred * 65535
    loss = K.mean(
                K.sum(
                        K.abs(y_true - y_pred) * A_i,
                        axis=(1,2,3)
                     )
                /
                K.sum(A_i, axis=(1,2,3))
            ) 
    return loss


def Masked_Root_Mean_Squared_Error(y_true, y_pred):
    '''Masked root mean squared error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    # original K.sqrt(K.mean(K.square(y_true - y_pred)))
    loss = K.sqrt(
            K.mean(
                    K.sum(
                            K.square(y_true - y_pred) * A_i,
                            axis=(1,2,3)
                         )
                    /
                    K.sum(A_i, axis=(1,2,3))
                  )
            )
    lower_boundary = K.less(y_pred, 0)
    lower_boundary = K.cast(lower_boundary, dtype='float32')
    upper_boundary = K.greater(y_pred, 65535)
    upper_boundary = K.cast(upper_boundary, dtype='float32')
    interval_loss = K.sum(lower_boundary * 10000 + upper_boundary * 10000)   
    return loss+interval_loss


def Masked_Root_Mean_Squared_Error_Sigmoid(y_true, y_pred):
    '''Masked root mean squared error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    # Since we are using a sigmoid activation function, scale the predictions from [0,1] to [0,65535]
    y_pred = y_pred * 65535
    # original K.sqrt(K.mean(K.square(y_true - y_pred)))
    loss = K.sqrt(
            K.mean(
                    K.sum(
                            K.square(y_true - y_pred) * A_i,
                            axis=(1,2,3)
                         )
                    /
                    K.sum(A_i, axis=(1,2,3))
                  )
            )
    lower_boundary = K.less(y_pred, 0)
    lower_boundary = K.cast(lower_boundary, dtype='float32')
    upper_boundary = K.greater(y_pred, 65535)
    upper_boundary = K.cast(upper_boundary, dtype='float32')
    interval_loss = K.sum(lower_boundary * 10000 + upper_boundary * 10000)   
    return loss+interval_loss


def Masked_Root_Mean_Squared_Error_Simple(y_true, y_pred):
    '''Masked root mean squared error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    # original K.sqrt(K.mean(K.square(y_true - y_pred)))
    loss = K.sqrt(
            K.mean(
                    K.sum(
                            K.square(y_true - y_pred) * A_i,
                            axis=(1,2,3)
                         )
                    /
                    K.sum(A_i, axis=(1,2,3))
                  )
            ) 
    return loss


def Masked_Root_Mean_Squared_Error_Simple_Sigmoid(y_true, y_pred):
    '''Masked root mean squared error custom loss function'''
    # create binary artifact maps from ground truth depth maps
    A_i = K.greater(y_true, 0)
    A_i = K.cast(A_i, dtype='float32')
    # Since we are using a sigmoid activation function, scale the predictions from [0,1] to [0,65535]
    y_pred = y_pred * 65535
    # original K.sqrt(K.mean(K.square(y_true - y_pred)))
    loss = K.sqrt(
            K.mean(
                    K.sum(
                            K.square(y_true - y_pred) * A_i,
                            axis=(1,2,3)
                         )
                    /
                    K.sum(A_i, axis=(1,2,3))
                  )
            ) 
    return loss


def berHu(c):
    '''Reverse Huber loss as stated in paper "Deeper Depth Prediction with Fully Convolutional Residual Networks" by Laina et al. and "The berhu
       penalty and the grouped effect" by L. Zwald and S. Lambert-Lacroix'''
    # does this current implementation makes sense? --> yes, it returns mae or mse
    # TODO implement this with binary mask too?
    def inverse_huber(y_true, y_pred):
        threshold = c * K.max(K.abs(y_true - y_pred))
        absolute_mean = K.mean(K.abs(y_true - y_pred))
        mask = K.less_equal(absolute_mean, threshold)
        mask = K.cast(mask, dtype='float32')
        return mask * absolute_mean + (1-mask) * K.mean(K.square(K.abs(y_true - y_pred)))
    return inverse_huber
        
    
class VGG:
    '''Class that contains building blocks for a residual VGG-like autoencoder network'''
    def __init__(self):
        self.layer_counting = {}
        
        
    def Block(self, number_of_layers, units, kernel_size, padding, activation, use_bn, momentum_bn):
        '''A block of <number_of_layers> convolutions with optional batch normalization added AFTER the non-linearity'''
        def Input(z):
            for i in range(1,number_of_layers+1):
                name = 'Conv' + str(kernel_size[0]) + '-' + str(units)
                # make sure we have unique layer names
                if name in self.layer_counting:
                    self.layer_counting[name] += 1
                else:
                    self.layer_counting[name] = 1
                name += '_' + str(self.layer_counting[name])
                z = Conv2D(filters=units, kernel_size=kernel_size, padding=padding, activation=activation, name=name)(z)
                if use_bn:
                    name_bn = name + '_BN'
                    z = BatchNormalization(name=name_bn, momentum=momentum_bn)(z)
            return z
        return Input
    
    
    def Residual_Downsampling_Block(self, units, kernel_size, padding, activation, use_bn, momentum_bn):
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
            z = Conv2D(filters=units, kernel_size=kernel_size, strides=(2,2), padding=padding, activation=activation, name=name)(z)
            if use_bn:
                name_bn = name + '_BN'
                z = BatchNormalization(name=name_bn, momentum=momentum_bn)(z)
            return z, skip
        return Input
    
    
    def Residual_Upsampling_Block(self, units, kernel_size, padding, activation, use_bn, momentum_bn):
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
            z = Conv2DTranspose(filters=units, kernel_size=kernel_size, strides=(2,2), padding="same", name=name)(z)
            z = Add(name=name_add)([z, skip])
            z = Activation(activation)(z)
            if use_bn:
                name_bn = name + '_BN'
                z = BatchNormalization(name=name_bn, momentum=momentum_bn)(z)
            return z
        return Input
    
    
    def Residual_Block(self, number_of_layers, units, kernel_size, padding, activation, use_bn, momentum_bn, skip_integration_mode='add'):
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
                z = Conv2D(filters=units, kernel_size=kernel_size, padding=padding)(z)
                if i == number_of_layers:
                    if skip_integration_mode.lower() == 'add':
                        z = Add(name=name_add)([z, skip])
                z = Activation(activation)(z)
                if use_bn:
                    name_bn = name + '_BN'
                    z = BatchNormalization(name=name_bn, momentum=momentum_bn)(z)
                if i == number_of_layers:
                    if skip_integration_mode.lower() == 'concat':
                        z = concatenate([skip, z], name='Concatenate_Skip_0')
            return z
        return Input


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description="Training of a VGG-style autoencoder for depth map prediction")
    Parser.add_argument("-t", "--train", type=str, default=None, help="Path to folder that contains the training and validation examples")
    Parser.add_argument("-x", "--output", type=str, default=None, help="Path to folder where all output is saved to")
    Parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size to train the network with")
    Parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs to train the network on")
    Parser.add_argument("--no_shuffle", default=False, action='store_true', help="Disables shuffling of batches for each epoch")
    Parser.add_argument("--no_scale", default=False, action='store_true', help="Disables scaling of input images to the range of [0,1]")
    Parser.add_argument("-o", "--optimizer", type=str, default="rmsprop", help="The optimizer to utilize for training. Supported are SGD, Adam and RMSprop.")
    Parser.add_argument("-p", "--periods", type=int, default=1, help="Number of epochs after which to save the current model (and its weights). 1 means every epoch.")
    Parser.add_argument("-d", "--decay", type=int, default=10, help="Reduce learning rate after every x epochs. Defaults to 10")
    Parser.add_argument("-f", "--factor_decay", type=float, default=0.5, help="Factor to reduce the learning rate. Defaults to 0.5")
    Parser.add_argument("--default_optimizers", default=False, action='store_true', help="Enable all keras optimizers, not only SGD, Adam and RMSprop. This will deactivate learning rate decay.")
    Parser.add_argument("--omit_batchnorm", default=False, action='store_true', help="Don't add batch normalization layers after convolutions.")
    Parser.add_argument("-m", "--momentum", type=float, default=0.99, help="Momentum used in batch normalization layers. Defaults to 0.99. If validation loss oscillates, try lowering it (e.g. to 0.6)")
    Parser.add_argument("--skip_0", type=str, default="add", help="Functionality of S0 skip connections. One of the following: 'add', 'concat', 'concat+' or 'disable'. Defaults to 'add'. 'Concat+' adds convolutions after concatenating.")
    Parser.add_argument("--sgd_momentum", type=str, default=None, help="Only works when using SGD optimizer: Not specified/'None': no momentum, 'normal': momentum with value from --sgd_momentum_value, 'nesterov': Use nesterov momentum with value from --sgd_momentum_value.")
    Parser.add_argument("--sgd_momentum_value", type=float, default=0.9, help="Only works when using SGD optimizer: Momentum value for SGD optimizer. Enable by using --sgd_momentum. Defaults to 0.9")
    Parser.add_argument("-l", "--loss", type=str, default="MMAE_simple", help="Loss function to utilize. Either MMAE MMAE_simple or MRMSE. Defaults to MMAE_simple")
    Parser.add_argument("--output_sigmoid_activation", type=str, default="", help="Adds an sigmoid activation function to the output layer. The provided argument defines whether the ground truth is scaled to also fit this interval ('scale_input') or if the predictions get scaled in the loss function ('scale_output'). Defaults to '' (no sigmoid activation function added)")
    args = Parser.parse_args()
    
    # training directory specified?
    if args.train is None:
        print("No directory with training examples specified!")
        print("For help use --help")
        exit()
        
    # does train folder exists?
    if not os.path.isdir(os.path.join(args.train, 'train')):
        print("Provided training directory contains no subfolder 'train'")
        exit()
        
    # valid batch size?
    if args.batch_size <= 0:
        print("Invalid batch size supplied!")
        exit()
        
    # valid epochs?
    if args.epochs <= 0:
        print("Invalid number of epochs supplied!")
        exit()
        
        
    # output directory specified?
    if args.output is None:
        print("No output directory specified!")
        print("For help use --help")
        exit()
        
        
    loss_func = None
    loss = args.loss.lower()
    output_sigmoid_activation = args.output_sigmoid_activation.lower()
    if loss == "mrmse":
        if output_sigmoid_activation == 'scale_output':
            print('Using masked-root-mean-squared-error sigmoid loss function')
            loss_func = Masked_Root_Mean_Squared_Error_Sigmoid
        else:
            print("Using masked-root-mean-squared-error loss function")
            loss_func = Masked_Root_Mean_Squared_Error
    elif loss == "mmae":
        if output_sigmoid_activation == 'scale_output':
            print('Using masked-mean-absolute-error sigmoid loss function')
            loss_func = Masked_Mean_Absolute_Error_Sigmoid
        else:
            print("Using masked-mean-absolute-error loss function")
            loss_func = Masked_Mean_Absolute_Error
    elif loss == 'mmae_simple':
        if output_sigmoid_activation == 'scale_output':
            print('Using masked-mean-absolute-error simple sigmoid loss function')
            loss_func = Masked_Mean_Absolute_Error_Simple_Sigmoid
        else:
            print("Using masked-mean-absolute-error simple loss function")
            loss_func = Masked_Mean_Absolute_Error_Simple
    elif loss == 'mrmse_simple':
        if output_sigmoid_activation == 'scale_output':
            print('Using masked-root-mean-squared-error simple sigmoid loss function')
            loss_func = Masked_Root_Mean_Squared_Error_Simple_Sigmoid
        else:
            print('Using masked-root-mean-squared-error simple loss function')
            loss_func = Masked_Root_Mean_Squared_Error_Simple
    else:
        print("Provided loss function is invalid. Defaulting to MMAE_simple with no sigmoid")
        loss_func = Masked_Mean_Absolute_Error_Simple
        output_sigmoid_activation = ''
        
    # skip connection 0 arguments
    s0_arg = args.skip_0.lower()
    if s0_arg != 'add' and s0_arg != 'concat' and s0_arg != 'disable' and s0_arg != 'concat+':
        print("Invalid argument for '--skip_0': " + s0_arg)
        print("Defaulting to 'add'")
        s0_arg = 'add'
        
    # create output directory
    os.makedirs(args.output, exist_ok=True)
    # create folder for logs
    log_dir = os.path.join(args.output, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # create folder for intermediate models
    model_dir = os.path.join(args.output, 'models')
    os.makedirs(model_dir, exist_ok=True)
    # create folder for figures
    figure_dir = os.path.join(args.output, 'figures')
    os.makedirs(figure_dir, exist_ok=True)        
        
    schedule = None
    optimizer = None
    if not args.default_optimizers:
        if args.optimizer.lower() == 'adam':
            print("Using adam optimizer")
            optimizer = Adam(lr=0.001)
            schedule = StepDecay(initAlpha=0.001, factor=args.factor_decay, dropEvery=args.decay)
        
        elif args.optimizer.lower() == 'rmsprop':
            print("Using RMSprop optimizer")
            optimizer = RMSprop(lr=0.001)
            schedule = StepDecay(initAlpha=0.001, factor=args.factor_decay, dropEvery=args.decay)
        
        elif args.optimizer.lower() == 'sgd':
            # check for momentum:
            if args.sgd_momentum is None:
                print("Using SGD optimizer without momentum")
                optimizer = SGD(lr=0.01)
            else:
                if args.sgd_momentum.lower() == 'normal':
                    print("Using normal SGD momentum = " + str(args.sgd_momentum_value))
                    optimizer = SGD(lr=0.01, momentum=args.sgd_momentum_value, nesterov=False)
                elif args.sgd_momentum.lower() == 'nesterov':
                    print("Using nesterov SGD momentum = " + str(args.sgd_momentum_value))
                    optimizer = SGD(lr=0.01, momentum=args.sgd_momentum_value, nesterov=True)
                else:
                    print("Unknown --sgd_momentum value. Defaulting to None")
                    optimizer = SGD(lr=0.01)
            schedule = StepDecay(initAlpha=0.01, factor=args.factor_decay, dropEvery=args.decay)

        if optimizer is None:
            print("Unsupported optimizer provided. If you want to use this unsupported optimizer, provide --custom False")
            print("For help use --help")
            exit()
    else:
        optimizer = args.optimizer
    
    scale_ground_truth = False
    if output_sigmoid_activation == 'scale_input':
        scale_ground_truth = True
    
    training_generator = DataGenerator(
            path_to_data_set=os.path.join(args.train, 'train'),
            batch_size=args.batch_size,
            image_size=(480,640),
            shuffle=not args.no_shuffle,
            scale_images=not args.no_scale,
            scale_ground_truth=scale_ground_truth
        )

    validation_generator = None
    if os.path.isdir(os.path.join(args.train, 'validation')):
        validation_generator = DataGenerator(
                path_to_data_set=os.path.join(args.train, 'validation'),
                batch_size=args.batch_size,
                image_size=(480,640),
                shuffle=not args.no_shuffle,
                scale_images=not args.no_scale,
                scale_ground_truth=scale_ground_truth
            )
        print("Using validation data")
    else:
        print("Not using validation data, because none where specified")
    
    current_date = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    current_model_name = 'model_' + current_date + '_epoch_{epoch:04d}.h5'
    checkpoint_path = os.path.join(model_dir, current_model_name)
    current_log_name = 'log_' + current_date
    log_path = os.path.join(log_dir, current_log_name)

    #checkpoint_callback = ModelCheckpoint(
    #        filepath=checkpoint_path,
    #        verbose=1,
    #        save_best_only=False,
    #        save_weights_only=False,
    #        mode='auto',
    #        period=args.periods)

    tensorboard_callback = TensorBoard(
            log_dir=log_path,
            histogram_freq = 0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=True,
            update_freq="epoch")

    #callback_list = [checkpoint_callback, tensorboard_callback]
    callback_list = [tensorboard_callback]
    if schedule is not None:
        callback_list.append(LearningRateScheduler(schedule))
    
    vgg = VGG()
    # Color branch
    input_color = Input(shape=(480,640,3), name="Color_Input")
    x = Model(inputs=input_color, outputs=input_color)
    
    # Infrared branch
    input_ir = Input(shape=(480,640,1), name="Infrared_Input")
    y = Model(inputs=input_ir, outputs=input_ir)

    # combine both branches
    combined = concatenate([x.output, y.output], name="Concatenate_Input")

    # zeroth skip connection start --> to transfer original input images to the end of the network
    skip_zero = combined

    # VGG16 style encoder (configuration D)

    z = vgg.Block(number_of_layers=2, units=64, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(combined)
    # max pooling replaced with strided convolution + first skip connection start
    z, skip_one = vgg.Residual_Downsampling_Block(units=64, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    z = vgg.Block(number_of_layers=2, units=128, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)
    # max pooling replaced with strided convolution + second skip connection start
    z, skip_two = vgg.Residual_Downsampling_Block(units=128, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    z = vgg.Block(number_of_layers=3, units=256, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)
    # max pooling replaced with strided convolution + third skip connection start
    z, skip_three = vgg.Residual_Downsampling_Block(units=256, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)
    # max pooling replaced with strided convolution + fourth skip connection start
    z, skip_four = vgg.Residual_Downsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)
    # max pooling replaced with strided convolution + fifth skip connection start
    z, skip_five = vgg.Residual_Downsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)

    # end of encoder part

    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)

    # start of decoder part (= mirrored encoder part)

    # upsampling with deconvolution + fifth skip connection target
    z = vgg.Residual_Upsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z, skip_five)
    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    # upsampling with deconvolution + fourth skip connection target
    z = vgg.Residual_Upsampling_Block(units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z, skip_four)
    z = vgg.Block(number_of_layers=3, units=512, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    # upsampling with deconvolution + third skip connection target
    z = vgg.Residual_Upsampling_Block(units=256, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z, skip_three)
    z = vgg.Block(number_of_layers=3, units=256, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    # upsampling with deconvolution + second skip connection target
    z = vgg.Residual_Upsampling_Block(units=128, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z, skip_two)
    z = vgg.Block(number_of_layers=2, units=128, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)


    # upsampling with deconvolution + first skip connection target
    z = vgg.Residual_Upsampling_Block(units=64, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z, skip_one)
    z = vgg.Block(number_of_layers=2, units=64, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)

    # end of decoder part
    
    if s0_arg == 'add':
        z = vgg.Residual_Block(number_of_layers=1, units=4, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum, skip_integration_mode='add')(z, skip_zero)
    elif s0_arg == 'concat' or s0_arg == 'concat+':
        z = vgg.Residual_Block(number_of_layers=1, units=4, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum, skip_integration_mode='concat')(z, skip_zero)
        if s0_arg == 'concat+':
            z = vgg.Block(number_of_layers=2, units=4, kernel_size=(3,3), padding="same", activation="relu", use_bn=not args.omit_batchnorm, momentum_bn=args.momentum)(z)

    # output layer
    if output_sigmoid_activation == '':
        z = Conv2D(1, kernel_size=(3,3), padding="same", name="Conv_Output")(z)
    else:
        z = Conv2D(1, kernel_size=(3,3), padding="same", activation='sigmoid', name="Conv_Output")(z)

    model = Model(inputs=[x.input, y.input], outputs=z)

    model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=['mae', 'mse', Masked_Mean_Absolute_Error, Masked_Root_Mean_Squared_Error, "accuracy", berHu(0.2), Masked_Mean_Absolute_Error_Simple])
    
    hist = model.fit_generator(
           generator=training_generator,
           validation_data=validation_generator,
           epochs=args.epochs,
           callbacks=callback_list)
    
    # plot the learning rate and loss
    N = np.arange(1, args.epochs+1)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, hist.history["loss"], label="train_loss")
    plt.plot(N, hist.history["val_loss"], label="val_loss")
    plt.title("Training Loss with Optimizer: " + args.optimizer)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(figure_dir,args.optimizer + '_' + str(args.epochs) + '_metrics'))
    
    if schedule is not None:
        N = np.arange(0, args.epochs)
        schedule.plot(N)
        plt.savefig(os.path.join(figure_dir,args.optimizer + '_' + str(args.epochs) + '_lr'))
    
    
    pickle.dump(hist.history, open(os.path.join(args.output,'history.p'), 'wb'))
    
    model.save(os.path.join(args.output,'final_model.h5'))
