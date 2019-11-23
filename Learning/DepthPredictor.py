# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:46:13 2019

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
import math
import matplotlib.pyplot as plt
import cv2

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

# The Predictor Class itself
class DepthPredictor:
    def __init__(self, path_to_model, model_uses_custom_loss, old_model):
        self.color = []
        self.infrared = []
        self.depth = []
        self.model = self.__load_model(path_to_model, model_uses_custom_loss, old_model)
        self.image_write_counter = 1
        
        
    def __load_model(self, path, model_uses_custom_loss, old_model):
        if os.path.isfile(path):
            if model_uses_custom_loss:
                loss_name = ""
                if old_model:
                    loss_name="Binary_Mean_Absolut_Error"
                else:
                    loss_name="Masked_Mean_Absolute_Error"
                return load_model(path, custom_objects={loss_name: Masked_Mean_Absolute_Error, 'inverse_huber':berHu(0.2), 'Masked_Root_Mean_Squared_Error':Masked_Root_Mean_Squared_Error})
            else:
                return load_model(path)
        else:
            print('Unable to load specified model: No such file!')
            return None
        
        
    def __cut_artifacts(self, images, threshold):
        '''Since the network outputs high values for artifacts, replace all pixels greater than the threshold with 0'''
        images[images>threshold]=0
        return images
    
    
    def __calc_histograms(self, images, cumulative_histogram):
        '''Creates histograms for the given images. Also adds each histogram to the given histogram (used for cumulative histogram)''' 
        n = []
        m = []
        for i in range(0,images.shape[0]):
            hist = cv2.calcHist([images[i].reshape(480,640)], [0], None, [65536], [0,65535])
            if images.dtype == np.float32:
                # predictions_raw
                hist_2 = cv2.calcHist([images[i].reshape(480,640)], [0], None, [65536], [int(np.amin(images[i])), int(np.amax(images[i]))])
                m.append(hist_2)
            if cumulative_histogram is not None:
                cumulative_histogram += hist
            n.append(hist)
        return np.asarray(n, dtype=np.float32), m
        
    
    def __predict(self, color, infrared):
        '''Predicts depth for the given color and infrared input. If process is true, clips and casts the prediction.'''
        color = color.reshape(-1, 480,640,3)
        infrared = infrared.reshape(-1, 480,640,1)
        predictions = self.model.predict([color, infrared])
        predictions_proc = np.clip(predictions, 0, 65535).astype(np.uint16)
        return predictions_proc, predictions            
    
    
    def PredictSingleImage(self, output_dir, input_color, input_infrared, input_depth=None, scale_images=True, threshold_offset=2000, differences_threshold=0.05, depth_scale=-1.0):
        '''Predicts a single depth image from given color and infrared input. Ground truth depth map is optional.'''
        self.image_write_counter = 1
        color = cv2.imread(input_color, cv2.IMREAD_COLOR)
        color_original = color.copy()
        infrared = cv2.imread(input_infrared, cv2.IMREAD_ANYDEPTH)
        infrared_original = infrared.copy()
        depth = cv2.imread(input_depth, cv2.IMREAD_ANYDEPTH)
        
        if color is None or infrared is None:
            print('Could not open provided color or infrared images!')
            return
        if scale_images == True:
            color = (color/255.).astype(np.float32)
            infrared = (infrared/65535.).astype(np.float32)
            
        # The prediction itself
        prediction, predictions_raw = self.__predict(color, infrared)
            
        threshold = -1
        # if ground truth available, utilize this ground truth to calculate threshold --> we can use normalization
        if depth is not None:
            threshold = np.amax(depth) + threshold_offset
            prediction = self.__cut_artifacts(prediction, threshold)
            depth = depth.reshape(-1, 480, 640)
        # if not, we need to fall back to histogram equalization
        prediction_normalized = self.__normalize(prediction, [threshold!=-1])
        ground_truth_depth_normalized = self.__normalize(depth, [threshold!=-1])
        
        # colorize normalized images
        prediction_colorized = self.__colorize(prediction_normalized, 2)
        ground_truth_depth_colorized = self.__colorize(ground_truth_depth_normalized, 2)
        
        # process other images
        infrared_colorized = cv2.cvtColor((infrared_original/256).astype(np.uint8), cv2.COLOR_GRAY2BGR).reshape(1, 480, 640, 3)
        
        # build an image with the colorizations, normalizations and input images
        summary = self.__create_overview_images(
                    prediction_normalized=prediction_normalized,
                    prediction_colorized=prediction_colorized,
                    ground_truth_normalized=ground_truth_depth_normalized,
                    ground_truth_colorized=ground_truth_depth_colorized,
                    color=color_original.reshape(1, 480, 640, 3),
                    infrared_colorized=infrared_colorized)
        
        # build image that contains not normalized depth images
        depth_not_normalized = self.__create_unprocessed_depth_images(
                    predicted_depth=prediction.reshape(-1, 480,640),
                    ground_truth_depth=depth)
        
        # calculate histograms
        hist_infrared = self.__calc_histograms(infrared_original.reshape(1, 480, 640, 1), None)[0]
        hist_depth_ground_truth = None
        if depth is not None:
            hist_depth_ground_truth = self.__calc_histograms(depth.reshape(1,480,640,1), None)[0]
        hist_depth_predicted = self.__calc_histograms(prediction.reshape(1, 480, 640, 1), None)[0]
        hist_depth_predicted_raw, hist_depth_predicted_raw_float = self.__calc_histograms(predictions_raw.reshape(1,480,640,1), None)
        
        differences = None
        if depth is not None:
            differences = self.__calc_differences(
                    ground_truth=depth,
                    predictions=prediction.reshape(1,480,640),
                    threshold_meters=differences_threshold,
                    depth_scale=depth_scale)
        
        self.__write_images(
                summaries=summary,
                depth=depth_not_normalized,
                predictions_unprocessed=prediction,
                histograms_infrared=hist_infrared,
                histograms_ground_truth_depth=hist_depth_ground_truth,
                histograms_predicted=hist_depth_predicted,
                histograms_predicted_raw=hist_depth_predicted_raw,
                histograms_predicted_raw_float=hist_depth_predicted_raw_float,
                differences=differences,
                output_dir=output_dir)        
        
        print('Successfully predicted and saved!')
        return
    
    
    def PredictImagesBatchWise(self, path_to_images, output_dir, batch_size, scale_images=True, threshold_offset=2000, differences_threshold=0.05, depth_scale=-1.0):
        '''Because the previous methods consume a large amount of memory, this function implements those functions batch wise'''
        self.image_write_counter = 1
        path_color = os.path.join(path_to_images, 'Color')
        path_infrared = os.path.join(path_to_images, 'Infrared')
        path_depth = os.path.join(path_to_images, 'Depth')
        if not os.path.isdir(path_color) or not os.path.isdir(path_infrared):
            print("Could not find folders " + path_color + " or " + path_infrared)
            exit()      
        files = os.listdir(path_color)
        available_files = [Path(file).stem for file in files if Path(file).suffix == '.jpg']
        number_batches = math.ceil(len(available_files)/batch_size)
        batch_counter = 1
        cumulative_histogram_infrared = np.zeros((65536, 1), dtype=np.float32)
        cumulative_histogram_depth = np.zeros((65536, 1), dtype=np.float32)
        cumulative_histogram_predictions_processed = np.zeros((65536, 1), dtype=np.float32)
        cumulative_histogram_predictions_raw = np.zeros((65536, 1), dtype=np.float32)
        while(len(available_files) > 0):
            print('Batch ' + str(batch_counter) + '/' + str(number_batches))
            current_batch = available_files[:batch_size]
            del available_files[:batch_size]
            # load images
            color, color_original, infrared, infrared_original, depth = self.__load_images_batch(path_color=path_color, path_infrared=path_infrared, path_depth=path_depth, batch=current_batch, scale_images=scale_images)
            # predict images
            predictions, predictions_raw = self.__predict(color, infrared)
            # create map that shows us which ground truth depth images are available
            sums = np.sum(depth, axis=(1,2,3))
            is_valid_depth = np.greater(sums, 0)
            
            # if ground truth depth available, use it to calculate threshold
            for i in range(0, predictions.shape[0]):
                if is_valid_depth[i]:
                    threshold = np.amax(depth[i]) + threshold_offset
                    predictions[i] = self.__cut_artifacts(predictions[i], threshold)
                
            # normalize
            predictions_normalized = self.__normalize(predictions, is_valid_depth)
            ground_truth_depth_normalized = self.__normalize(depth, is_valid_depth)
            
            # colorize normalized images
            predictions_colorized = self.__colorize(predictions_normalized, 2)
            ground_truth_depth_colorized = self.__colorize(ground_truth_depth_normalized, 2)
            
            # process infrared to be able to display it
            infrared_colorized = []
            for i in range(0,infrared_original.shape[0]):
                buffer = cv2.cvtColor((infrared_original[i]/256).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                infrared_colorized.append(buffer)
            infrared_colorized = np.asarray(infrared_colorized, dtype=np.uint8)
            
            # build image with colorization, normalization and input images
            summary = self.__create_overview_images(
                    prediction_normalized=predictions_normalized,
                    prediction_colorized=predictions_colorized,
                    ground_truth_normalized=ground_truth_depth_normalized,
                    ground_truth_colorized=ground_truth_depth_colorized,
                    color=color_original,
                    infrared_colorized=infrared_colorized)
            
            # build image that contains not normalized depth images
            depth_not_normalized = self.__create_unprocessed_depth_images(
                    predicted_depth=predictions.reshape(-1, 480, 640),
                    ground_truth_depth=depth.reshape(-1, 480, 640))
            
            hist_infrared = self.__calc_histograms(infrared_original.reshape(-1, 480, 640, 1), cumulative_histogram_infrared)[0]
            hist_depth_ground_truth = self.__calc_histograms(depth.reshape(-1, 480, 640, 1), cumulative_histogram_depth)[0]
            hist_depth_predicted = self.__calc_histograms(predictions.reshape(-1, 480 ,640, 1), cumulative_histogram_predictions_processed)[0]
            hist_depth_predicted_raw, hist_depth_predicted_raw_float = self.__calc_histograms(predictions_raw.reshape(-1, 480, 640, 1), cumulative_histogram_predictions_raw)
            
            differences = self.__calc_differences(
                    ground_truth=depth.reshape(-1, 480, 640),
                    predictions=predictions.reshape(-1, 480, 640),
                    threshold_meters=differences_threshold,
                    depth_scale=depth_scale)
            
            self.__write_images(
                    summaries=summary,
                    depth=depth_not_normalized,
                    predictions_unprocessed=predictions,
                    histograms_infrared=hist_infrared,
                    histograms_ground_truth_depth=hist_depth_ground_truth,
                    histograms_predicted=hist_depth_predicted,
                    histograms_predicted_raw=hist_depth_predicted_raw,
                    histograms_predicted_raw_float=hist_depth_predicted_raw_float,
                    differences=differences,
                    output_dir=output_dir)
                    
            batch_counter += 1
        # write cumulative histograms
        self.__write_cumulative_histograms(
                histogram_infrared=cumulative_histogram_infrared,
                histogram_ground_truth_depth=cumulative_histogram_depth,
                histogram_predicted=cumulative_histogram_predictions_processed,
                histogram_predicted_raw=cumulative_histogram_predictions_raw,
                output_dir=output_dir)
        print('Successfully predicted and saved!')
        return
    
    
    def __colorize(self, images, color_scheme=2):  
        '''Colorizes the given images based on the specified color scheme'''
        if images is None:
            return None
        n = []           
        for img in images:
            buffer = cv2.applyColorMap(img, color_scheme)
            n.append(buffer)    
        return np.asarray(n, dtype=np.uint8).reshape(-1, 480, 640, 3)
    
    
    def __normalize(self, images, utilize_normalize):
        '''Normalizes the 16-bit depth images to 8-bit utilizing either histogram equalization or normalization
        utilize_normalize is list that contains for every image in images True if normalization should be utilized and False if histogram equalization should be utilized'''
        if images is None:
            return None
        n = []
        for i in range(0,images.shape[0]):
            if utilize_normalize[i]:
                buffer = cv2.normalize(images[i], None, 0, 65535, cv2.NORM_MINMAX)
                buffer = (buffer/256).astype(np.uint8)
                n.append(buffer)
            else:
                buffer = (images[i]/256).astype(np.uint8)
                buffer = cv2.equalizeHist(buffer)
                n.append(buffer)                           
        return np.asarray(n, dtype=np.uint8).reshape(-1, 480, 640)
    
    
    def __create_overview_images(self, prediction_normalized, prediction_colorized, ground_truth_normalized, ground_truth_colorized, color, infrared_colorized):
        '''Stacks all images into a single image'''
        n = []
        for idx, img in enumerate(prediction_normalized):
            prediction = np.concatenate((cv2.cvtColor(prediction_normalized[idx], cv2.COLOR_GRAY2BGR), prediction_colorized[idx]), axis=0)
            ground_truth = None
            if ground_truth_normalized is not None and ground_truth_colorized is not None:
                ground_truth = np.concatenate((cv2.cvtColor(ground_truth_normalized[idx], cv2.COLOR_GRAY2BGR), ground_truth_colorized[idx]), axis=0)
            inp = np.concatenate((color[idx], infrared_colorized[idx]), axis=0)
            
            comb = None
            if ground_truth is not None:
                comb = np.concatenate((inp, ground_truth, prediction), axis=1)
            else:
                comb = np.concatenate((inp, prediction), axis=1)
            n.append(comb)
        return np.asarray(n, dtype=np.uint8)
        
    
    def __create_unprocessed_depth_images(self, predicted_depth, ground_truth_depth):
        '''Stacks images into a single image'''
        n = []
        for idx, img in enumerate(predicted_depth):
            comb = None
            if ground_truth_depth is not None:
                    comb = np.concatenate((ground_truth_depth[idx], predicted_depth[idx]), axis=1)
            else:
                comb = predicted_depth[idx]
            n.append(comb)
        return np.asarray(n, dtype=np.uint16)
                
                
    def __write_images(self, summaries, depth, predictions_unprocessed, histograms_infrared, histograms_ground_truth_depth, histograms_predicted, histograms_predicted_raw, histograms_predicted_raw_float, differences, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        for i in range(0, summaries.shape[0]):
            filename_summary = str(self.image_write_counter) + '_summary.jpg'
            cv2.imwrite(os.path.join(output_dir, filename_summary), summaries[i])
            filename_prediction = str(self.image_write_counter) + '_predicted_depth.png'
            cv2.imwrite(os.path.join(output_dir, filename_prediction), predictions_unprocessed[i])
            
            if depth is not None:
                filename_depth = str(self.image_write_counter) + '_depth_not_normalized.png'
                cv2.imwrite(os.path.join(output_dir, filename_depth), depth[i])
            if differences is not None:
                filename_differences = str(self.image_write_counter) + '_differences.jpg'
                cv2.imwrite(os.path.join(output_dir, filename_differences), differences[i])
            # create histograms
            # Infrared solo
            fig, axs = plt.subplots(2,1, sharex=True)
            axs[0].plot(histograms_infrared[i])
            axs[0].set_title('Linear Infrared')
            axs[0].set_yscale('linear')
            axs[1].plot(histograms_infrared[i])
            axs[1].set_title('Log Infrared')
            axs[1].set_yscale('log')
            for ax in axs.flat:
                ax.set(xlabel='Values', ylabel='Occurences')        
            #for ax in axs.flat:
                #ax.label_outer()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,str(self.image_write_counter) + '_histograms_infrared.png'))
            
            if histograms_ground_truth_depth is not None:
                # grount truth vs processed predicted
                fig, axs = plt.subplots(2,2, sharex=True)
                axs[0,0].plot(histograms_ground_truth_depth[i])
                axs[0,0].set_title('Linear Ground Truth')
                axs[0,0].set_yscale('linear')
                axs[1,0].plot(histograms_ground_truth_depth[i])
                axs[1,0].set_title('Log Ground Truth')
                axs[1,0].set_yscale('log')
                axs[0,1].plot(histograms_predicted[i])
                axs[0,1].set_title('Linear Processed Predicted')
                axs[0,1].set_yscale('linear')
                axs[1,1].plot(histograms_predicted[i])
                axs[1,1].set_title('Log Processed Predicted')
                axs[1,1].set_yscale('log')
                for ax in axs.flat:
                    ax.set(xlabel='Values', ylabel='Occurences')        
                #for ax in axs.flat:
                    #ax.label_outer()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir,str(self.image_write_counter) + '_histograms_ground_truth_predicted_processed.png'))
            
                # ground truth vs raw predicted
                fig, axs = plt.subplots(2,2, sharex=True)
                axs[0,0].plot(histograms_ground_truth_depth[i])
                axs[0,0].set_title('Linear Ground Truth')
                axs[0,0].set_yscale('linear')
                axs[1,0].plot(histograms_ground_truth_depth[i])
                axs[1,0].set_title('Log Ground Truth')
                axs[1,0].set_yscale('log')
                axs[0,1].plot(histograms_predicted_raw[i])
                axs[0,1].set_title('Linear Raw Predicted')
                axs[0,1].set_yscale('linear')
                axs[1,1].plot(histograms_predicted_raw[i])
                axs[1,1].set_title('Log Raw Predicted')
                axs[1,1].set_yscale('log')
                for ax in axs.flat:
                    ax.set(xlabel='Values', ylabel='Occurences')        
                #for ax in axs.flat:
                    #ax.label_outer()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir,str(self.image_write_counter) + '_histograms_ground_truth_predicted_raw.png'))
                
                # ground truth solo
                fig, axs = plt.subplots(2,1 , sharex=True)
                axs[0].plot(histograms_ground_truth_depth[i])
                axs[0].set_title('Linear Ground Truth')
                axs[0].set_yscale('linear')
                axs[1].plot(histograms_ground_truth_depth[i])
                axs[1].set_title('Log Ground Truth')
                axs[1].set_yscale('log')
                for ax in axs.flat:
                    ax.set(xlabel='Values', ylabel='Occurences')        
                #for ax in axs.flat:
                    #ax.label_outer()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir,str(self.image_write_counter) + '_histograms_ground_truth.png'))
            
            # predicted processed vs raw
            fig, axs = plt.subplots(2,2, sharex=True)
            axs[0,0].plot(histograms_predicted[i])
            axs[0,0].set_title('Linear Processed Predicted')
            axs[0,0].set_yscale('linear')
            axs[1,0].plot(histograms_predicted[i])
            axs[1,0].set_title('Log Processed Predicted')
            axs[1,0].set_yscale('log')
            axs[0,1].plot(histograms_predicted_raw[i])
            axs[0,1].set_title('Linear Raw Predicted')
            axs[0,1].set_yscale('linear')
            axs[1,1].plot(histograms_predicted_raw[i])
            axs[1,1].set_title('Log Raw Predicted')
            axs[1,1].set_yscale('log')
            for ax in axs.flat:
                ax.set(xlabel='Values', ylabel='Occurences')        
            #for ax in axs.flat:
                #ax.label_outer()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,str(self.image_write_counter) + '_histograms_predicted_processed_predicted_raw.png'))
            
            # predicted raw float solo
            fix, axs = plt.subplots(2,1, sharex=True)
            axs[0].plot(histograms_predicted_raw_float[i])
            axs[0].set_title('Linear Raw Predicted No Clipping')
            axs[0].set_yscale('linear')
            axs[1].plot(histograms_predicted_raw_float[i])
            axs[1].set_title('Log Raw Predicted No Clipping')
            axs[1].set_yscale('log')
            for ax in axs.flat:
                ax.set(xlabel='Values', ylabel='Occurences')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, str(self.image_write_counter) + '_histogram_predicted_raw_no_clipping.png'))
            
            plt.close('all')
                
            self.image_write_counter += 1
        
    
    def __calc_differences(self, ground_truth, predictions, threshold_meters, depth_scale):
        '''Visualizes the differences between ground truth and predictions based on the provided threshold'''
        n = []
        if depth_scale == -1.0:
            return np.asarray(n, dtype=np.uint8)
        artifacts = np.equal(ground_truth, 0)
        for i in range(0, ground_truth.shape[0]):
            diff = np.abs(ground_truth[i] - predictions[i])
            threshold_pixel_domain = threshold_meters / depth_scale
            mask = np.less_equal(diff, threshold_pixel_domain)
            col = np.zeros((480,640,3), dtype=np.uint8)
            col[mask==0] = (0,0,255)
            col[mask==1] = (0,255,0)
            col[artifacts[i]==1] = (0,0,0)
            n.append(col)
            
        return np.asarray(n, dtype=np.uint8)
    
    def __write_cumulative_histograms(self, histogram_infrared, histogram_ground_truth_depth, histogram_predicted, histogram_predicted_raw, output_dir):
        '''Creates graphs of the given histogram arrays and writes them on disk'''
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(histogram_infrared)
        axs[0].set_title('Linear Cumulative Infrared Histogram')
        axs[0].set_yscale('linear')
        axs[1].plot(histogram_infrared)
        axs[1].set_title('Log Cumulative Infrared Histogram')
        axs[1].set_yscale('log')
        for ax in axs.flat:
            ax.set(xlabel='Values', ylabel='Occurences')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_infrared_histogram.png'))
        
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(histogram_ground_truth_depth)
        axs[0].set_title('Linear Cumulative Ground Truth Histogram')
        axs[0].set_yscale('linear')
        axs[1].plot(histogram_ground_truth_depth)
        axs[1].set_title('Log Cumulative Ground Truth Histogram')
        axs[1].set_yscale('log')
        for ax in axs.flat:
            ax.set(xlabel='Values', ylabel='Occurences')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_ground_truth_histogram.png'))
        
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(histogram_predicted)
        axs[0].set_title('Linear Cumulative Processed Predicted Histogram')
        axs[0].set_yscale('linear')
        axs[1].plot(histogram_predicted)
        axs[1].set_title('Log Cumulative Processed Predicted Histogram')
        axs[1].set_yscale('log')
        for ax in axs.flat:
            ax.set(xlabel='Values', ylabel='Occurences')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_processed_predicted_histogram.png'))
        
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(histogram_predicted_raw)
        axs[0].set_title('Linear Cumulative Raw Predicted Histogram')
        axs[0].set_yscale('linear')
        axs[1].plot(histogram_predicted_raw)
        axs[1].set_title('Log Cumulative Raw Predicted Histogram')
        axs[1].set_yscale('log')
        for ax in axs.flat:
            ax.set(xlabel='Values', ylabel='Occurences')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_raw_predicted_histogram.png'))
        plt.close('all')
        

    
    def __load_images_batch(self, path_color, path_infrared, path_depth, batch, scale_images=True):
        color = []
        color_original = []
        infrared = []
        infrared_original = []
        depth = []
        batch_size = len(batch)
        for file in batch:
            img_c = cv2.imread(os.path.join(path_color, file+'.jpg'), cv2.IMREAD_COLOR)
            img_i = cv2.imread(os.path.join(path_infrared, file+'.png'), cv2.IMREAD_ANYDEPTH)
            img_d = cv2.imread(os.path.join(path_depth, file+'.png'), cv2.IMREAD_ANYDEPTH)
            color_original.append(img_c)
            infrared_original.append(img_i)
            if img_d is None:
                img_d = np.zeros((480,640,1), dtype=np.uint16)
            if scale_images:
                img_c = (img_c/255).astype(np.float32)
                img_i = (img_i/65535).astype(np.float32)
            color.append(img_c)
            infrared.append(img_i)
            depth.append(img_d)
        return np.asarray(color).reshape(batch_size, 480, 640, 3), np.asarray(color_original).reshape(batch_size, 480, 640, 3), np.asarray(infrared).reshape(batch_size, 480, 640, 1), np.asarray(infrared_original).reshape(batch_size, 480, 640, 1), np.asarray(depth).reshape(batch_size, 480, 640, 1)
                
 
if __name__ == '__main__':   
    Parser = argparse.ArgumentParser(description="For a specified set of infrared and color images, predict the corresponding depth image, colorize it and store it in the provided folder.")
    Parser.add_argument("-f", "--folder", type=str, default=None, help="If multiple depth images should be predicted, the color and infrared images, along with optional ground truth depth images, should be placed in a folder with subfolders 'Color', 'Infrared' and optionally 'Depth'. This is the path to this folder.")
    Parser.add_argument("-c", "--color", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding color image.")
    Parser.add_argument("-i", "--infrared", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding infrared image.")
    Parser.add_argument("-g", "--ground_truth", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding ground truth depth image. Can be ignored if no ground truth is available.")
    #Parser.add_argument("-v", "--visualization_mode", type=str, default="normalize", help="The visualization mode that should be utilized for depth image visualization. Can be either 'normalize' or 'histogram'.")
    Parser.add_argument("-b", "--batch_size", type=int, default=1, help="When multiple depth images should be predicted, this is the batch size that should be utilized while predicting.")
    Parser.add_argument("-m", "--model", type=str, default=None, help="Path to the model that should be utilized for predicting depth images.")
    Parser.add_argument("-o", "--output", type=str, default=None, help="Path to where the predicted images should be saved to.")
    Parser.add_argument("--no_scaling", default=False, action='store_true', help="Don't scale the input images to the range [0,1]")
    Parser.add_argument("--default_loss", default=False, action='store_true', help="Use the default mean absolute error loss funtion")
    #Parser.add_argument("--swap_artifacts", default=False, action='store_true', help="Swap artifacts in predicted depth images from 65535 to 0")
    #Parser.add_argument("-p", "--prefix", type=str, default=None, help="Prefix for the image save filename")
    Parser.add_argument("-t", "--threshold_offset", type=int, default=2000, help="Offset for depth image normalization. Defaults to 2000. Only utilized if ground truth is given.")
    #Parser.add_argument("--no_processing", default=False, action='store_true', help="Predicted depth image's range is not clipped to 16 bit. Can result in strange behavior.")
    Parser.add_argument("--old_model", default=False, action='store_true', help="For old models, the loss function was called binary mean absolut error. Activate this if an 'Unknown loss function' error is thrown.")
    #Parser.add_argument("--force_histogram", default=False, action='store_true', help="When normalizing depth images, force histogram equalization even if ground truth depth is available")
    Parser.add_argument("-d", "--difference_threshold", type=float, default=0.05, help="Utilized for difference visualization of ground truth and prediction. Maximum difference between ground truth and prediction in meters which is considered ok. Defaults to 0.05")
    Parser.add_argument("--depth_scale_text", type=str, default=None, help="Text file containing depth scale of the utilized depth camera. Alternatively, use --depth_scale_value to directly provide a float")
    Parser.add_argument("--depth_scale_value", type=float, default=-1.0, help="Depth scale of the utilized depth camera. Alternatively, provide text file containing this scale with --depth_scale_text")
    # TODO implement force_histogram option

    args = Parser.parse_args()
    
    # necessary parameters
    if not args.model:
        print("No model supplied.")
        print("For help use --help")
        exit()
        
    if not os.path.isfile(args.model):
        print("The supplied model is no file!")
        print("Make sure to supply the correct path to the model.")
        exit()
        
    if not args.output:
        print("No output path specified.")
        print("For help use --help")
        exit()
        
    depth_scale = -1.0
    if args.depth_scale_text is None:
        # utilize direct value
        depth_scale = args.depth_scale_value
    else:
        if os.path.isfile(args.depth_scale_text):
            with open(args.depth_scale_text, 'r') as file:
                depth_scale = float(file.readline())
                
    # check if something went wrong
    if depth_scale == -1.0:
        print('Unable to infer correct camera depth scale. Please specify a readable text file containing the scale or directly provide the depth scale')
        print('Will not visualize differences between ground truth and predictions')
        print('For help, utilize --help')
        
    time_start = cv2.getTickCount()  
    
    # create depth predictor
    dp = DepthPredictor(args.model, model_uses_custom_loss=not args.default_loss, old_model=args.old_model)
    
    # load images
    if args.folder is None:
        print('Single image prediction mode.')
        dp.PredictSingleImage(
                output_dir=args.output,
                input_color=args.color,
                input_infrared=args.infrared,
                input_depth=args.ground_truth,
                scale_images=not args.no_scaling,
                threshold_offset=args.threshold_offset,
                differences_threshold=args.difference_threshold,
                depth_scale=depth_scale)
    else:
        dp.PredictImagesBatchWise(
                path_to_images=args.folder,
                output_dir=args.output,
                batch_size=args.batch_size,
                scale_images=not args.no_scaling,
                threshold_offset=args.threshold_offset,
                differences_threshold=args.difference_threshold, 
                depth_scale=depth_scale)
    
    
    
    time_end = cv2.getTickCount()
    time_overall = (time_end - time_start) / cv2.getTickFrequency()
    
    print("Finished predicting all input images after " + str(time_overall) + " seconds.")
    exit()