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

# the custom loss function
def Masked_Mean_Absolut_Error(y_true, y_pred):
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
                return load_model(path, custom_objects={loss_name: Masked_Mean_Absolut_Error})
            else:
                return load_model(path)
        else:
            print('Unable to load specified model: No such file!')
            return None
        
        
    def __cut_artifacts(self, images, threshold):
        '''Since the network outputs high values for artifacts, replace all pixels greater than the threshold with 0'''
        images[images>threshold]=0
        return images
    
    
    def __get_histogram(self, image, title, scale='linear'):
        '''Creates a histogram for the specified image and returns it'''
        image = image.reshape(1,480,640,1)
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2)
        x = []
        values = 0
        if image.dtype == np.uint16:
            x = np.arange(65536)
            values=65536
        elif image.dtype == np.uint8:
            x = np.arange(256)
            values=256
        else:
            print("Unsupported image depth!")
            return
        y = cv2.calcHist(image, [0], None, [values], [0,values])
        #fig.suptitle(title)
        ax1.plot(x, y.ravel())
        ax1.set(ylabel='Occurences', yscale=scale)
        ax1.set_title(title1)
        ax1.label_outer()
        ax2.plot(x, y.ravel())
        ax2.set(xlabel='Values', ylabel='Occurences', yscale=scale)
        ax2.set_title(title2)
        return fig        
        
    
    def __predict(self, color, infrared, process=True):
        '''Predicts depth for the given color and infrared input. If process is true, clips and casts the prediction.'''
        color = color.reshape(-1, 480,640,3)
        infrared = infrared.reshape(-1, 480,640,1)
        predictions = self.model.predict([color, infrared])
        if process:
            predictions = np.clip(predictions, 0, 65535).astype(np.uint16)
        return predictions            
    
    
    def PredictSingleImage(self, output_dir, filename_prefix, input_color, input_infrared, input_depth=None, scale_images=True, threshold_offset=2000, process_prediction=True):
        '''Predicts a single depth image from given color and infrared input. Ground truth depth map is optional.'''
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
        prediction = self.__predict(color, infrared, process_prediction)
            
        threshold = -1
        # if ground truth available, utilize this ground truth to calculate threshold --> we can use normalization
        if depth is not None:
            threshold = np.amax(depth) + threshold_offset
            prediction = self.__cut_artifacts(prediction, threshold)
        # if not, we need to fall back to histogram equalization
        prediction_normalized = self.__normalize(prediction, threshold!=-1)
        ground_truth_depth_normalized = self.__normalize(depth, threshold!=-1)
        
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
        
        # build image that contains unprocessed depth images
        if depth is not None:
            depth = depth.reshape(-1, 480,640)
        unprocessed_depth = self.__create_unprocessed_depth_images(
                    predicted_depth=prediction.reshape(-1, 480,640),
                    ground_truth_depth=depth)
        
        
        self.__write_combinations(
                summaries=summary,
                depth=unprocessed_depth,
                output_dir=output_dir,
                prefix=filename_prefix)        
        
        print('Successfully predicted and saved!')
        return
    
    
    def LoadImages(self, path_to_images=None, color=None, infrared=None, depth=None, additive_load=False, normalize_images=True):
        if not additive_load:
            self.color = []
            self.infrared = []
            self.depth = []        
        if path_to_images is None:
            # single image prediction
            if os.path.isfile(color) and os.path.isfile(infrared):
                img_c = cv2.imread(color, cv2.IMREAD_COLOR)
                img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
                img_i = cv2.imread(infrared, cv2.IMREAD_ANYDEPTH)
                if normalize_images:
                    img_c = (img_c/255.).astype(np.float32)
                    img_i = (img_i/65535.).astype(np.float32)
                img_d = None
                if depth is not None and os.path.isfile(depth):
                    img_d = cv2.imread(depth, cv2.IMREAD_ANYDEPTH)                 
                self.color.append(img_c)
                self.infrared.append(img_i)
                self.depth.append(img_d)               
        else:
            # multi image prediction
            if os.path.isdir(path_to_images):
                path_color = os.path.join(path_to_images, 'Color')
                path_infrared = os.path.join(path_to_images, 'Infrared')
                path_depth = os.path.join(path_to_images, 'Depth')          
                if os.path.isdir(path_color) and os.path.isdir(path_infrared):
                    available_files = os.listdir(path_color)
                    files = [file for file in available_files if Path(file).suffix == '.jpg']
                    for file in files:
                        file_name = Path(file).stem
                        img_c = cv2.imread(os.path.join(path_color,file), cv2.IMREAD_COLOR)
                        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)    
                        img_i = cv2.imread(os.path.join(path_infrared, file_name + '.png'), cv2.IMREAD_ANYDEPTH)
                        if normalize_images:
                            img_c = (img_c/255.).astype(np.float32)
                            img_i = (img_i/65535.).astype(np.float32)
                        img_d = None                       
                        if os.path.isdir(path_depth):
                            img_d = cv2.imread(os.path.join(path_depth, file_name + '.png'), cv2.IMREAD_ANYDEPTH)               
                        self.color.append(img_c)
                        self.infrared.append(img_i)
                        self.depth.append(img_d)
        self.color = np.asarray(self.color, dtype=np.float32).reshape(-1,480,640,3)
        self.infrared = np.asarray(self.infrared, dtype=np.float32).reshape(-1,480,640,1)
        self.depth = np.asarray(self.depth, dtype=np.float32).reshape(-1,480,640,1)
        return self.color.shape[0]
    
    
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
        '''Normalizes the 16-bit depth images to 8-bit utilizing either histogram equalization or normalization'''
        if images is None:
            return None
        n = []
        if utilize_normalize:
            for img in images:
                buffer = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                buffer = (buffer/256).astype(np.uint8)
                n.append(buffer)
        else:
            for img in images:
                buffer = (img/256).astype(np.uint8)
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
                
    
    
    def __create_images(self, pred_colorized, pred_depth_n, pred_depth_raw, ground_colorized, ground_depth_n, ground_depth_raw):
        img_summary = []
        img_diff = []      
        for idx, img in enumerate(pred_colorized):
            # difference image
            b_diff = None
            if ground_depth_raw.shape[0] != 0:
                b_diff_d = np.abs(pred_depth_raw[idx] - ground_depth_raw[idx])
                b_diff_color = np.abs(img - ground_colorized[idx])
                b_diff = np.concatenate((cv2.cvtColor(b_diff_d, cv2.COLOR_GRAY2BGR), b_diff_color), axis=0)
            img_diff.append(b_diff)           
            # comparison images
            summary = np.concatenate((cv2.cvtColor(pred_depth_n[idx], cv2.COLOR_GRAY2BGR), img), axis=0)
            if ground_colorized.shape[0] != 0:
                ground = np.concatenate((cv2.cvtColor(ground_depth_n[idx], cv2.COLOR_GRAY2BGR), ground_colorized[idx]), axis=0)
                summary = np.concatenate((summary, ground), axis=1)
            img_summary.append(summary)
        return np.asarray(img_summary, dtype=np.uint8), np.asarray(img_diff, dtype=np.uint8)    
    
    
    def __write_images(self, summaries, differences, save_path, batch_size):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)      
        for idx, img in enumerate(summaries):
            filename_summary = str(idx) + '_summary.jpg'    
            cv2.imwrite(os.path.join(save_path, filename_summary), img)
        for idx, img in enumerate(differences):
            if img is not None:
                filename_difference = str(idx) + '_difference.jpg'
                cv2.imwrite(os.path.join(save_path, filename_difference), img)
                
                
    def __write_combinations(self, summaries, depth, output_dir, prefix):
        for idx, img in enumerate(summaries):
            filename = ""
            if prefix is not None:
                filename = prefix + '_'
            filename_summary = filename + 'summary.jpg'
            cv2.imwrite(os.path.join(output_dir, filename_summary), summaries[idx])
            
            if depth is not None:
                filename_depth = filename + 'original_depth.png'
                cv2.imwrite(os.path.join(output_dir, filename_depth), depth[idx])
          
            
    def __write_images_batch(self, summaries, differences, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for i in range(0,summaries.shape[0]):
            filename_summary = str(self.image_write_counter) + '_summary.jpg'
            filename_difference = str(self.image_write_counter) + '_difference.jpg'
            cv2.imwrite(os.path.join(save_path, filename_summary), summaries[i])
            cv2.imwrite(os.path.join(save_path, filename_difference), differences[i])
            self.image_write_counter += 1
            
            
    def PredictImages(self, batch_size):
        # predict
        pred_depth = self.model.predict([self.color, self.infrared], batch_size=batch_size).reshape(-1, 480,640)     
        return pred_depth
        
    
    def ProcessImages(self, predictions, visualization_mode='normalize'):
        # colorize prediction
        pred_color, ground_color = self.__colorize(predictions, self.depth)
        # normalize depth images for visualization
        pred_depth_n, ground_depth_n = self.__normalize(predictions, self.depth, normalization_mode = visualization_mode)
        # create images
        summaries, differences = self.__create_images(
                pred_color = pred_color, 
                pred_depth_n = pred_depth_n,
                pred_depth_raw = predictions,
                ground_color = ground_color,
                ground_depth_n = ground_depth_n,
                ground_depth_raw = self.depth.reshape(-1, 480, 640))       
        return summaries, differences
    
    
    def WriteImages(self, summaries, differences, save_path):
        # write images
        self.__write_images(summaries=summaries, differences=differences, save_path=save_path)
        
        
    def __load_images_batch(self, path_color, path_infrared, path_depth, batch, normalize_inputs):
        color = []
        infrared = []
        depth = []
        batch_size = len(batch)
        for file in batch:
            img_c = cv2.imread(os.path.join(path_color, file+'.jpg'), cv2.IMREAD_COLOR)
            img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
            img_i = cv2.imread(os.path.join(path_infrared, file+'.png'), cv2.IMREAD_ANYDEPTH)
            img_d = cv2.imread(os.path.join(path_depth, file+'.png'), cv2.IMREAD_ANYDEPTH)
            if img_d is None:
                img_d = np.zeros((480,640,1), dtype=np.uint16)
            if normalize_inputs:
                img_c = (img_c/255).astype(np.float32)
                img_i = (img_i/65535).astype(np.float32)
            color.append(img_c)
            infrared.append(img_i)
            depth.append(img_d)
        return np.asarray(color).reshape(batch_size, 480, 640, 3), np.asarray(infrared).reshape(batch_size, 480, 640, 1), np.asarray(depth).reshape(batch_size, 480, 640, 1)
            
        
    def __process_images_batch(self, predictions, ground_truth, color, infrared, visualization_mode, images_scaled):
        # colorize predictions
        pred_colorized, ground_colorized = self.__colorize(predictions, ground_truth, color_scheme=2)
        # normalize depth images for visualization
        pred_depth_n, ground_depth_n = self.__normalize(predictions, ground_truth, visualization_mode)
        # create images
        summaries, differences = self.__create_images(
                pred_color=pred_colorized,
                pred_depth_n=pred_depth_n,
                pred_depth_raw=predictions,
                ground_color=ground_colorized,
                ground_depth_n=ground_depth_n,
                ground_depth_raw=ground_truth.reshape(-1, 480, 640))
        return summaries, differences
    
    
    def PredictImagesBatchWise(self, path_to_images, save_path, batch_size, visualization_mode="normalize", normalize_inputs=True):
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
        while(len(available_files) > 0):
            print('Batch ' + str(batch_counter) + '/' + str(number_batches))
            current_batch = available_files[:batch_size]
            del available_files[:batch_size]
            # load images
            color, infrared, depth = self.__load_images_batch(path_color=path_color, path_infrared=path_infrared, path_depth=path_depth, batch=current_batch, normalize_inputs=normalize_inputs)
            # predict images
            pred_raw = self.model.predict([color, infrared]).reshape(-1, 480, 640)
            # process images
            summaries, differences = self.__process_images_batch(pred_raw, depth, color, infrared, visualization_mode, normalize_inputs)
            # write images
            self.__write_images_batch(summaries, differences, save_path)
            batch_counter += 1
                
 
if __name__ == '__main__':   
    Parser = argparse.ArgumentParser(description="For a specified set of infrared and color images, predict the corresponding depth image, colorize it and store it in the provided folder.")
    Parser.add_argument("-f", "--folder", type=str, default=None, help="If multiple depth images should be predicted, the color and infrared images, along with optional ground truth depth images, should be placed in a folder with subfolders 'Color', 'Infrared' and optionally 'Depth'. This is the path to this folder.")
    Parser.add_argument("-c", "--color", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding color image.")
    Parser.add_argument("-i", "--infrared", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding infrared image.")
    Parser.add_argument("-g", "--ground_truth", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding ground truth depth image. Can be ignored if no ground truth is available.")
    Parser.add_argument("-v", "--visualization_mode", type=str, default="normalize", help="The visualization mode that should be utilized for depth image visualization. Can be either 'normalize' or 'histogram'.")
    Parser.add_argument("-b", "--batch_size", type=int, default=1, help="When multiple depth images should be predicted, this is the batch size that should be utilized while predicting.")
    Parser.add_argument("-m", "--model", type=str, default=None, help="Path to the model that should be utilized for predicting depth images.")
    Parser.add_argument("-o", "--output", type=str, default=None, help="Path to where the predicted images should be saved to.")
    Parser.add_argument("--no_scaling", default=False, action='store_true', help="Don't scale the input images to the range [0,1]")
    Parser.add_argument("--default_loss", default=False, action='store_true', help="Use the default mean absolute error loss funtion")
    Parser.add_argument("--swap_artifacts", default=False, action='store_true', help="Swap artifacts in predicted depth images from 65535 to 0")
    Parser.add_argument("-p", "--prefix", type=str, default=None, help="Prefix for the image save filename")
    Parser.add_argument("-t", "--threshold_offset", type=int, default=2000, help="Offset for depth image normalization. Defaults to 2000. Only utilized if ground truth is given.")
    Parser.add_argument("--no_processing", default=False, action='store_true', help="Predicted depth image's range is not clipped to 16 bit. Can result in strange behavior.")
    Parser.add_argument("--old_model", default=False, action='store_true', help="For old models, the loss function was called binary mean absolut error. Activate this if an 'Unknown loss function' error is thrown.")
    Parser.add_argument("--force_histogram", default=False, action='store_true', help="When normalizing depth images, force histogram equalization even if ground truth depth is available")
    # TODO implement force_histogram option
    # TODO implement multi image prediction possibility

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
        
    time_start = cv2.getTickCount()  
    
    # create depth predictor
    dp = DepthPredictor(args.model, model_uses_custom_loss=not args.default_loss, old_model=args.old_model)
    
    # load images
    if args.folder is None:
        print('Single image prediction mode.')
        dp.PredictSingleImage(
                output_dir=args.output,
                filename_prefix=args.prefix,
                input_color=args.color,
                input_infrared=args.infrared,
                input_depth=args.ground_truth,
                scale_images=not args.no_scaling,
                threshold_offset=args.threshold_offset,
                process_prediction=not args.no_processing
                )
    else:
        dp.PredictImagesBatchWise(
                path_to_images=args.folder,
                save_path=args.output,
                batch_size=args.batch_size,
                visualization_mode=args.visualization_mode,
                normalize_inputs=not args.no_scaling)
    
    
    
    time_end = cv2.getTickCount()
    time_overall = (time_end - time_start) / cv2.getTickFrequency()
    
    print("Finished predicting all input images after " + str(time_overall) + " seconds.")
    exit()