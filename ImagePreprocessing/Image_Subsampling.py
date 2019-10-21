#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:13:55 2019

@author: Julien Fischer
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
from pathlib import Path

def process_images(files,                   # file names of all bag files
                   path,                    # path to where the bag files is located
                   save_path,               # path to where to save the sampled images
                   subsample_factor,        # subsampling factor
                   verbose=0):              # verbosity
    
    Frames_overall = 0
    
    frame_index = 1
    
    # create train folder with Color, Infrared and Depth folders
    save_train = os.path.join(save_path + 'train')
    save_color = os.path.join(save_train, 'Color')
    save_ir = os.path.join(save_train, 'Infrared')
    save_depth = os.path.join(save_train, 'Depth')
    os.makedirs(save_train, exist_ok=True)
    os.makedirs(save_color, exist_ok=True)
    os.makedirs(save_ir, exist_ok=True)
    os.makedirs(save_depth, exist_ok=True)
    
    pipeline = rs.pipeline()
    config = rs.config()
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
    for file in tqdm(files):        
        time_start_one = cv2.getTickCount()      
        
        filepath = os.path.join(path, file)
        rs.config.enable_device_from_file(config, filepath, False)
            
        pipeline.start(config)
        device = pipeline.get_active_profile().get_device()
        
        playback = device.as_playback()
        playback.set_real_time(False)
            
        Success = True
        frame_counter = -1  # used to implement subsample functionality
        
        while Success:
            Success, frames = pipeline.try_wait_for_frames()
            if Success is True:
                frame_counter += 1
                Frames_overall += 1
                if frame_counter % subsample_factor == 0:
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    ir_frame = frames.get_infrared_frame()
                
                
                    # tell numpy explicitly what datatype we want (not necessary)
                
                    color_image = np.asarray(color_frame.get_data(), dtype=np.uint8)
                    ir_image = np.asarray(ir_frame.get_data(), dtype=np.uint16)
                    depth_image_raw = np.asarray(depth_frame.get_data(), dtype=np.uint16)
                    depth_image = cv2.medianBlur(depth_image_raw, 7)
                
                    # convert all color images to BGR so they get adequately saved as RGB / infrared and depth image only have one channel, so they don't matter (imwrite assumes bgr images and saves them as rgb images)
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                    # save images to respective folders
                    color_filepath = os.path.join(save_color, str(frame_index) + ".jpg")
                    ir_filepath = os.path.join(save_ir, str(frame_index) + ".png")
                    depth_filepath = os.path.join(save_depth, str(frame_index) + ".png")
                    
                    frame_index += 1
                
                    cv2.imwrite(color_filepath, color_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    cv2.imwrite(ir_filepath, ir_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    cv2.imwrite(depth_filepath, depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
        pipeline.stop()
        time_end_one = cv2.getTickCount()
        if verbose > 1:
            time_one = (time_end_one - time_start_one) / cv2.getTickFrequency()
            print("Needed " + str(time_one) + " seconds to extract all images from " + file)

    return Frames_overall               
        

# object for parsing command line
Parser = argparse.ArgumentParser(description="Reads all recorded bag files from the provided folder and saves the included images to the provided output folder.")
Parser.add_argument("-i", "--input", type=str, help="Path to folder that contains the scenes that should be subsampled")
Parser.add_argument("-o", "--output", type=str, help="Path where the subsampled images should be saved to")
Parser.add_argument("-1", "--skip_test", type=str, help="The first scene that should not be sampled from. This scene will be used as test data")
Parser.add_argument("-2", "--skip_validation", type=str, help="The second scene that should not be sampled from. This scene will be used as validation data")
Parser.add_argument("-s", "--subsample", type=int, default=10, help="The subsampling rate")
Parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity settings. 0 - no extra messages. 1 - basic debug messages. 2 - additional runtime messages")

args = Parser.parse_args()

if not args.input:
    print("No input parameters have been given.")
    print("For help type --help")
    exit()
if not args.output:
    print("No output parameters have been given.")
    print("For help type --help")
    exit()
if not args.skip_test or not args.skip_validation:
    print("No skip scenes defined. It is not possible to sample from all scenes!")
    print("For help type --help")
    exit()
    
if not args.subsample:
    print("No subsampling rate defined. Assuming 10")
    
time_start_overall = cv2.getTickCount()


# check if paths exist
if not os.path.isdir(args.input):
    print("Could not find the directory " + args.input)
    exit()

print("Searching for .bag files in " + args.input)

# gather all .bag files
available_files = os.listdir(args.input)

files = [file for file in available_files if Path(file).suffix == ".bag" and Path(file).stem != args.skip_test and Path(file).stem != args.skip_validation]
print("Found " + str(len(files)) + " .bag files:")
print(files)

with open(os.path.join(args.output, 'log.txt'), 'a') as f:
    f.write("Scenes that get subsampled:\n")
    for file in files:
        f.write(file+'\n')
    f.write("Scenes that get ignored:\n")
    f.write("Test: " + args.skip_test+'\n')
    f.write("Validation: " + args.skip_validation+'\n')

number_frames = 0

print("Starting preprocessing...")
number_frames += process_images(files, args.input, args.output, args.subsample, args.verbose)

time_end_overall = cv2.getTickCount()
time_overall = (time_end_overall - time_start_overall) / cv2.getTickFrequency()

print("Finished processing all " + str(number_frames) + " frames in " + str(time_overall) + " seconds.")


exit()