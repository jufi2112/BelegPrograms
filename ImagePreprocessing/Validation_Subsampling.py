#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:15:52 2019

@author: Julien Fischer
"""
import os
from pathlib import Path
import argparse

def Subsample(main_path, folder_color, folder_infrared, folder_depth, subsample_factor, target_path):
    path_color = os.path.join(main_path, folder_color)
    path_infrared = os.path.join(main_path, folder_infrared)
    path_depth = os.path.join(main_path, folder_depth)
    if not os.path.isdir(path_color) or not os.path.isdir(path_infrared) or not os.path.isdir(path_depth):
        print("Provided subfolders do not exist!")
        return
    available_files = os.listdir(path_color)
    files = [Path(file).stem for file in available_files if Path(file).suffix == '.jpg']
    
    target_color = os.path.join(target_path, folder_color)
    target_infrared = os.path.join(target_path, folder_infrared)
    target_depth = os.path.join(target_path, folder_depth)
    
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    if not os.path.isdir(target_color):
        os.mkdir(target_color)
    if not os.path.isdir(target_infrared):
        os.mkdir(target_infrared)
    if not os.path.isdir(target_depth):
        os.mkdir(target_depth)
    
    counter = 1
    for idx, file in enumerate(files):
        if idx % subsample_factor == 0:
            os.rename(os.path.join(path_color, file+'.jpg'), os.path.join(target_color, str(counter)+'.jpg'))
            os.rename(os.path.join(path_infrared, file+'.png'), os.path.join(target_infrared, str(counter)+'.png'))
            os.rename(os.path.join(path_depth, file+'.png'), os.path.join(target_depth, str(counter)+'.png'))
            counter += 1
    print("Finished")
    return
            

if __name__ == '__main__':
    Parser = argparse.ArgumentParser(description="Subsamples images in the provided folder by a specified factor")
    Parser.add_argument("-f", "--folder_input", type=str, default=None, help="The original folder whos images should be subsampled")
    Parser.add_argument("-o", "--output", type=str, default=None, help="The location where the subsampled files should be placed")
    Parser.add_argument("-s", "--subsample", type=int, default=10, help="The subsampling factor. Defaults to 10.")
    Parser.add_argument("-c", "--color", type=str, default="Color", help="Name of the folder that contains the color images. Defaults to 'Color'")
    Parser.add_argument("-i", "--infrared", type=str, default="Infrared", help="Name of the folder that contains the infrared images. Defaults to 'Infrared'")
    Parser.add_argument("-d", "--depth", type=str, default="Depth", help="Name of the folder that contains the depth images. Defaults to 'Depth'")
    args = Parser.parse_args()
    
    if not args.folder_input or not args.output:
        print("Missing necessary parameters.")
        print("For help use --help")
        exit()
        
    if not os.path.isdir(args.folder_input):
        print("Provided input folder does not exist!")
        exit()
        
    Subsample(args.folder_input, args.color, args.infrared, args.depth, args.subsample, args.output)
    
    print('Subsampling finished!')