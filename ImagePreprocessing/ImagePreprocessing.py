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
import matplotlib.pyplot as plt

def read_images(files, path, save_path, downsample_filter_magnitude=1):
    # configure pipeline
    #try:
    pipeline = rs.pipeline()
    config = rs.config()
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
    for file in tqdm(files):
        # skip file if it is marked as defect
        if file.split("_")[0] == "DEFECT":
            continue
        
        filepath = os.path.join(path, file)
        rs.config.enable_device_from_file(config, filepath, False)
            
        pipeline.start(config)
        device = pipeline.get_active_profile().get_device()
        
        # retrieve the depth scale so we can transform the depth frame's values to meters
        sensor = device.first_depth_sensor()
        depth_scale = sensor.get_depth_scale()
        
        playback = device.as_playback()
        playback.set_real_time(False)
        
        # initialize decimation filter
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, downsample_filter_magnitude)
            
        Success = True
        Frames = 0
        
        # create folder structure to save extracted images to
        
        # first, check if we have outdoor or indoor lighting
        path_separated = path.split("/")
        if len(path_separated) == 0:
            print("Error when trying to split " + path + " on '/': No splitting possible!")
            exit()
                
        save_path = os.path.join(save_path, path_separated[-1])
                
        # next, get the scene the .bag file contains and create respective subfolder
        file_split = file.split("_")
        if len(file_split) == 0:
            print("Error when trying to split " + file + " on '_': No splitting possible!")
            exit()
        
        save_path = os.path.join(save_path, file_split[0] + "_" + file_split[1])
        
        
        # create subfolder if not existant
        os.makedirs(save_path, exist_ok=True)
                
        
            
        while Success:
            Success, frames = pipeline.try_wait_for_frames()
                #print(Success)
            if Success is True:
                Frames += 1
                    #print(Frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame()
                
                # tell numpy explicitly what datatype we want (not necessary)
                
                color_image = np.asarray(color_frame.get_data(), dtype=np.uint8)
                ir_image = np.asarray(ir_frame.get_data(), dtype=np.uint8)
                depth_image_raw = np.asarray(depth_frame.get_data(), dtype=np.uint16)
                
                depth_frame_proc = decimation.process(depth_frame)      
                depth_image_proc = np.asarray(depth_frame_proc.get_data(), dtype=np.uint16)
                
                # convert all color images to BGR so they get adequately saved / infrared and depth image only have one channel, so they don't matter
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # multiply depth image's values by the cameras scaling factor to retrieve absolut depth values
                depth_image_raw = depth_scale * depth_image_raw
                depth_image_proc = depth_scale * depth_image_proc
                
                # save images to respective folders
                
                # TODO test if scaling of depth image loses information
                # TODO think aboult folder structure for each scene (own folder for every image type?)

                
                    
                
                    
                
                #plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                #plt.imshow(ir_image, cmap="gray")
                #plt.imshow(depth_image, cmap="gray")
                #plt.pause(.1)
                #plt.draw()
        pipeline.stop()
                
                
    #finally:
        #pass
        

# object for parsing command line
Parser = argparse.ArgumentParser(description="Reads all recorded bag files from the provided folder and saves the included images to the provided output folder.")
Parser.add_argument("-i", "--input", type=str, help="Path to main folder that contains subfolders 'Outdoor_Lighting' and 'Indoor_Lighting' which contain the recorded bag files")
Parser.add_argument("-o", "--output", type=str, help="Path where the included images should be saved to")
Parser.add_argument("-d", "--decimation", type=int, default=1 ,help="Decimation filter magnitude. Values of 2 and 3 perform median downsampling, values greater than 3 perform mean downsampling")

args = Parser.parse_args()

if not args.input:
    print("No input parameters have been given.")
    print("For help type --help")
    exit()
if not args.output:
    print("No output parameters have been given.")
    print("For help type --help")
    exit()
    
if not args.decimation:
    print("No decimation filter magnitude input. Assuming 1 (no downsampling")
    args.decimation = 1 # necessary with default=1 ?
    
# paths to subfolders
outdoor_path = os.path.join(args.input, "Outdoor_Lighting")
indoor_path = os.path.join(args.input, "Indoor_Lighting")

# check if paths exist
if not os.path.isdir(outdoor_path):
    print("Could not find the directory " + outdoor_path)
    exit()
    
if not os.path.isdir(indoor_path):
    print("Could not find the directory " + indoor_path)
    exit()

print("Searching for .bag files in " + outdoor_path + " and " + indoor_path)

# gather all .bag files
outdoor_available_files = os.listdir(outdoor_path)
indoor_available_files = os.listdir(indoor_path)

outdoor_files = [file for file in outdoor_available_files if Path(file).suffix == ".bag"]
indoor_files = [file for file in indoor_available_files if Path(file).suffix == ".bag"]

print("Found " + str(len(outdoor_files)) + " outdoor .bag files")
print("Found " + str(len(indoor_files)) + " indoor .bag files")


print("Starting preprocessing for files in " + outdoor_path + "...")
read_images(outdoor_files, outdoor_path, args.decimation)

print("Starting preprocessing for files in " + indoor_path + "...")
read_images(indoor_files, indoor_path, args.decimation)

exit()