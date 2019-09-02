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

def read_images(files, path):
    # configure pipeline
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        for file in tqdm(files):
            filepath = os.path.join(path, file)
            rs.config.enable_device_from_file(config, filepath, False)
            
            pipeline.start(config)
            device = pipeline.get_active_profile().get_device()
            playback = device.as_playback()
            playback.set_real_time(False)
            
            Success = True
            Frames = 0
            
            while Success:
                (Success, frames) = pipeline.try_wait_for_frames()
                #print(Success)
                if Success is True:
                    Frames += 1
                    #print(Frames)
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    ir_frame = frames.get_infrared_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    ir_image = np.asanyarray(ir_frame.get_data())
                
                #plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                #plt.imshow(ir_image, cmap="gray")
                #plt.imshow(depth_image, cmap="gray")
                #plt.pause(.1)
                #plt.draw()
            pipeline.stop()
                
                
    finally:
        pass
        

# object for parsing command line
Parser = argparse.ArgumentParser(description="Reads all recorded bag files from the provided folder and saves the included images to the provided output folder.")
Parser.add_argument("-i", "--input", type=str, help="Path to main folder that contains subfolders 'Outdoor_Lighting' and 'Indoor_Lighting' which contain the recorded bag files")
Parser.add_argument("-o", "--output", type=str, help="Path where the included images should be saved to")

args = Parser.parse_args()

if not args.input:
    print("No input parameters have been given.")
    print("For help type --help")
    exit()
if not args.output:
    print("No output parameters have been given.")
    print("For help type --help")
    exit()
    
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
read_images(outdoor_files, outdoor_path)

print("Starting preprocessing for files in " + indoor_path + "...")
read_images(indoor_files, indoor_path)

exit()