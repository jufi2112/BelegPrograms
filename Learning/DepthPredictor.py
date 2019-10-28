# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:46:13 2019

@author: Julien Fischer
"""

import os
from keras.models import load_model
import cv2
from pathlib import Path
import numpy as np
import argparse

# The Predictor Class itself
class DepthPredictor:
    def __init__(self, path_to_model):
        self.color = []
        self.infrared = []
        self.depth = []
        self.model = self.__load_model(path_to_model)
        
    def __load_model(self, path):
        if os.path.isfile(path):
            return load_model(path)
        else:
            print('Unable to load specified model: No such file!')
            return None
        
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
    
    def __colorize(self, d1, d2, color_scheme=2):
        
        n1 = []
        n2 = []
        
        #b1 = np.zeros(d1.shape, dtype=np.uint16)
        
        for img in d1:
            buffer = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
            buffer = 255 - (buffer/256.).astype(np.uint8)
            buffer = cv2.equalizeHist(buffer)
            buffer = cv2.applyColorMap(buffer, color_scheme)
            n1.append(buffer)
            
        
        for img in d2:
            buffer = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
            buffer = 255 - (buffer/256.),astype(np.uint8)
            buffer = cv2.equalizeHist(buffer)
            buffer = cv2.applyColorMap(buffer, color_scheme)    
            n2.append(buffer)
        
        return np.asarray(n1, dtype=np.uint8).reshape(-1, 480, 640, 3), np.asarray(n2, dtype=np.uint8).reshape(-1, 480, 640, 3)
    
    def __normalize(self, d1, d2, normalization_mode="normalize"):
        
        n1 = []
        n2 = []
        if normalization_mode == "normalize":
            for img in d1:
                buffer = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                buffer = (buffer/256).astype(np.uint8)
                n1.append(buffer)
            for img in d2:
                buffer = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                buffer = (buffer/256).astype(np.uint8)
                n2.append(buffer)
        elif normalization_mode == "histogram":
            for img in d1:
                buffer = (img/256).astype(np.uint8)
                buffer = cv2.equalizeHist(buffer)
                n1.append(buffer)
            for img in d2:
                buffer = (img/256).astype(np.uint8)
                buffer = cv2.equalizeHist(buffer)
                n2.append(buffer)
                
        return np.asarray(n1, dtype=np.uint8).reshape(-1, 480, 640), np.asarray(n2, dtype=np.uint8).reshape(-1, 480, 640)        
        
    def __create_images(self, pred_color, pred_depth_n, pred_depth_raw, ground_color, ground_depth_n, ground_depth_raw):
        img_summary = []
        img_diff = []
        
        for idx, img in enumerate(pred_color):
            # difference image
            b_diff = None
            if ground_depth_raw.shape[0] != 0:
                b_diff_d = pred_depth_raw[idx] - ground_depth_raw[idx]
                b_diff_color = img - ground_color[idx]
                b_diff = np.concatenate((cv2.cvtColor(b_diff_d, cv2.COLOR_GRAY2BGR), b_diff_color), axis=0)
            img_diff.append(b_diff)
            
            # comparison images
            summary = np.concatenate((cv2.cvtColor(pred_depth_n[idx], cv2.COLOR_GRAY2BGR), img), axis=0)
            if ground_color.shape[0] != 0:
                ground = np.concatenate((cv2.cvtColor(ground_depth_n[idx], cv2.COLOR_GRAY2BGR), ground_color[idx]), axis=0)
                summary = np.concatenate((summary, ground), axis=1)
            img_summary.append(summary)
            
        return np.asarray(img_summary, dtype=np.uint8).reshape(-1, 480, 640, 3), np.asarray(img_diff, dtype=np.uint8).reshape(-1, 480, 640, 3)
            
          
    def PredictImages(self, visualization_mode='normalize', batch_size):
        # predict
        depth_pred = self.model.predict([self.color, self.infrared], batch_size=batch_size).reshape(-1, 480,640)
        # colorize prediction
        color_pred, color_ground = self.__colorize(depth_pred, self.depth)
        # normalize depth images for visualization
        depth_pred_n, depth_ground_n = self.__normalize(depth_pred, self.depth, normalization_mode = visualization_mode)
        
        
        
        
        for imgset in self.images:
            img_c = imgset.Color
            img_i = imgset.Infrared
            img_d = imgset.Depth
            
            depth_pred = self.model.predict([(img_c.reshape(1,480,640,3)/255.).astype(np.float32), (img_i.reshape(1, 480, 640, 1)/65535.).astype(np.float32)])
            
            depth_pred = depth_pred.reshape(480,640).astype(np.uint16)
            color_pred, color_ground = self.__colorize(depth_pred, img_d)
            
            # normalize depth image for visualization
            if visualization_mode == 'normalize':
                pred_norm = np.zeros(depth_pred.shape, dtype=np.uint16)
                pred_norm = cv2.normalize(depth_pred, pred_norm, 0, 65535, cv2.NORM_MINMAX)
                pred_norm = (pred_norm/256.).astype(np.uint8)
            elif visualization_mode == 'histogram':
                pred_norm = (depth_pred/256.).astype(np.uint8)
                pred_norm = cv2.equalizeHist(pred_norm)
            
            img_pred = np.concatenate((cv2.cvtColor(pred_norm, cv2.COLOR_GRAY2BGR), color_pred), axis=0)
            img = img_pred
            
            if img_d is not None:
                # normalize depth image for visualization
                if visualization_mode == 'normalize':
                    ground_norm = np.zeros(img_d.shape, dtype=np.uint16)
                    ground_norm = cv2.normalize(img_d, ground_norm, 0, 65535, cv2.NORM_MINMAX)
                    ground_norm = (ground_norm/256.).astype(np.uint8)
                elif visualization_mode == 'histogram':
                    ground_norm = (img_d/256.).astype(np.uint8)
                    ground_norm = cv2.equalizeHist(ground_norm)
                
                img_ground = np.concatenate((cv2.cvtColor(ground_norm, cv2.COLOR_GRAY2BGR), color_ground), axis=0)
                img = np.concatenate((img_pred, img_ground), axis=1)
                
            list_images.append(img)
        return list_images
    
if __name__ == '__main__':
    
    Parser = argparse.ArgumentParser(description="For a specified set of infrared and color images, predict the corresponding depth image, colorize it and store it in the provided folder.")
    Parser.add_argument("-f", "--folder", type=str, default=None, help="If multiple depth images should be predicted, the color and infrared images, along with optional ground truth depth images, should be placed in a folder with subfolders 'Color', 'Infrared' and optionally 'Depth'. This is the path to this folder.")
    Parser.add_argument("-c", "--color", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding color image.")
    Parser.add_argument("-i", "--infrared", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding infrared image.")
    Parser.add_argument("-g", "--ground_truth", type=str, default=None, help="If only a single depth image should be predicted, this is the corresponding ground truth depth image. Can be ignored if no ground truth is available.")
    Parser.add_argument("-n", "--normalization_mode", type=str, default="normalize", help="The normalization mode that should be utilized for depth image visualization. Can be either 'normalize' or 'histogram'.")
    Parser.add_argument("-b", "--batch_size", type=int, default=1, help="When multiple depth images should be predicted, this is the batch size that should be utilized while predicting.")
    Parser.add_argument("-m", "--model", type=str, default=None, help="Path to the model that should be utilized for predicting depth images.")
    Parser.add_argument("-o", "--output", type=str, default=None, help="Path to where the predicted images should be saved to.")
    Parser.add_argument("-z", "--normalize_input", type=bool, default=True, help="Whether the input images should be normalized to the range from 0 to 1 or not.")
    
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
       
    # create depth predictor
    dp = DepthPredictor(args.model)
    
    # load images
    if args.folder is not None:
        dp.LoadImages(path_to_images=args.folder, normalize_images=args.normalize_input)
    else:
        dp.LoadImages(color=args.color, infrared=args.infrared, depth=args.ground_truth, normalize_images=args.normalize_input)
    
    # predict images
    dp.   