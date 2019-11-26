#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:49:58 2019

@author: Julien Fischer
"""
import sys
import warnings
if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=FutureWarning)
import os
from keras.models import load_model
import keras.backend as K
import cv2
from pathlib import Path
import numpy as np
import argparse
import cv2
import pyrealsense2 as rs


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


if __name__ == '__main__':
    Parser = argparse.ArgumentParser(description="Predicts depth from a streaming RealSense camera or from a recorded sequence")
    Parser.add_argument("-p", "--playback_path", type=str, default=None, help="Path to a recorded sequence that should be predicted. Defaults to None (i.e. streaming configuration)")
    Parser.add_argument("--no_realtime", default=False, action='store_true', help="Disables real time mode")
    Parser.add_argument("-m", "--model", type=str, default=None, help="Path to model.")
    Parser.add_argument("--scale_output", default=False, action='store_true', help="Scale the output of the network.")
    Parser.add_argument("--no_clip", default=False, action='store_true', help="Don't clip the output predictions to [0,65535]")
    args = Parser.parse_args()
    
    if args.model is None:
        print('No model specified.')
        print('Use --help for more information')
        exit()
    
    # load the model
    model = load_model(args.model, custom_objects={'Masked_Mean_Absolute_Error':Masked_Mean_Absolute_Error,
                                                   'Masked_Mean_Absolute_Error_Simple':Masked_Mean_Absolute_Error_Simple, 
                                                   'Masked_Mean_Absolute_Error_Simple_Sigmoid':Masked_Mean_Absolute_Error_Simple_Sigmoid,
                                                   'Masked_Mean_Absolute_Error_Sigmoid':Masked_Mean_Absolute_Error_Sigmoid,
                                                   'Masked_Root_Mean_Squared_Error':Masked_Root_Mean_Squared_Error,
                                                   'Masked_Root_Mean_Squared_Error_Simple':Masked_Root_Mean_Squared_Error_Simple,
                                                   'Masked_Root_Mean_Squared_Error_Simple_Sigmoid':Masked_Root_Mean_Squared_Error_Simple_Sigmoid,
                                                   'Masked_Root_Mean_Squared_Error_Sigmoid':Masked_Root_Mean_Squared_Error_Sigmoid,
                                                   'inverse_huber':berHu(0.2)})
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        if args.playback_path is not None:
            #print('Test')
            # playback mode
            if Path(args.playback_path).suffix != '.bag':
                print('Specified invalid rosbag file!. Make sure the target file is a .bag file')
                exit()
            rs.config.enable_device_from_file(config, args.playback_path)
        
            
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y16, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
        pipeline.start(config)
                
        #cv2.namedWindow("Color and Infrared", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("Depth and Colorization", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("Colorized Depth Stream", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("Predicted Depth Stream", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE) 
        #cv2.namedWindow("Original Images", cv2.WINDOW_AUTOSIZE)
        
        colorizer = rs.colorizer()
                
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            infrared_frame = frames.get_infrared_frame()
                
            depth_color_frame = colorizer.colorize(depth_frame)
                
                
            depth_color_image = cv2.cvtColor(np.asanyarray(depth_color_frame.get_data()), cv2.COLOR_RGB2BGR).astype(np.uint8)
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())         # is in RGB format
            infrared_image = np.asanyarray(infrared_frame.get_data())
                
            # predict images
            pred = model.predict([(color_image/255.).astype(np.float32).reshape(1, 480, 640, 3), (infrared_image/65535.).astype(np.float32).reshape(1, 480, 640, 1)])
                
            if args.scale_output:
                pred = pred * 65535
                    
            if not args.no_clip:
                pred = np.clip(pred, 0,65535)
                    
            pred = pred.astype(np.uint16).reshape(480,640)
            
            pred_edited = np.copy(pred)
            pred_edited[pred_edited > (np.amax(depth_image))] = 0
            pred_edited = cv2.normalize(pred_edited, None, 0, 65535, cv2.NORM_MINMAX)
            pred_edited = (pred_edited/256.).astype(np.uint8)
            pred_edited = 255 - pred_edited
            pred_edited = cv2.applyColorMap(pred_edited, 2)
            
            #pred_no_edit = np.copy(pred)
            #pred_no_edit = cv2.normalize(pred_no_edit, None, 0, 65535, cv2.NORM_MINMAX)
            #pred_no_edit = (pred_no_edit/256.).astype(np.uint8)
            #pred_no_edit = cv2.applyColorMap(pred_no_edit, 2)
            
            #pred_hist = np.copy(pred)
            #pred_hist = cv2.equalizeHist((pred_hist/256).astype(np.uint8))
            #pred_hist = 255 - pred_hist
            #pred_hist = cv2.applyColorMap(pred_hist, 2)
                
            # stack images
            comb_1 = np.concatenate((cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), cv2.cvtColor((infrared_image/256.).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.uint8)), axis=0)
            comb_2 = np.concatenate((cv2.cvtColor((depth_image/256.).astype(np.uint8), cv2.COLOR_GRAY2BGR), cv2.cvtColor(depth_color_image, cv2.COLOR_RGB2BGR)), axis=0)
            comb_3 = np.concatenate((cv2.cvtColor((pred/256.).astype(np.uint8), cv2.COLOR_GRAY2BGR), pred_edited), axis=0)
            #comb_4 = np.concatenate((pred_no_edit, pred_hist), axis=0)
            comb = np.concatenate((comb_1, comb_2, comb_3), axis=1)
            
            #cv2.imshow("Color and Infrared", comb_1)
            #cv2.imshow("Depth and Colorization", comb_2)
            #cv2.imshow("Depth Stream", depth_image)
            #cv2.imshow("Colorized Depth Stream", cv2.cvtColor(depth_color_image, cv2.COLOR_RGB2BGR))
            #cv2.imshow("Predicted Depth Stream", pred)
            cv2.imshow("Output", comb)
            #cv2.imshow("Original Images", comb_1)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
                
    finally:
        pass
    
