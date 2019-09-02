#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:38:55 2019

@author: Julien Fischer
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt

Parser = argparse.ArgumentParser(description="Reads the specified ROSBAG file and iterates one time through all frames, displaying the number of frames at the end.")
Parser.add_argument("-i", "--input", type=str, help="Path the ROSBAG file that should be read in.")

args = Parser.parse_args()

if not args.input:
    print("No input parameter has been given.")
    print("For help type --help")
    exit()
    
if not os.path.isfile(args.input):
    print("No such file: " + args.input)
    exit()
    
print("Starting to read the .bag file")
time_start = cv2.getTickCount()

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, args.input, False)

pipeline.start(config)
device = pipeline.get_active_profile().get_device()
playback = device.as_playback()
playback.set_real_time(False)

Success = True
Frames = 0
time_loading_end = cv2.getTickCount()
print("Finished loading, starting to iterate through the .bag file's frames")
while Success:
    Success, frames = pipeline.try_wait_for_frames()
    if Success is True:
        Frames += 1
        #color_frame = frames.get_color_frame()
        #infrared_frame = frames.get_infrared_frame()
        #depth_frame = frames.get_depth_frame()
        
        # explicitly tell numpy what datatype we want (not necessary)
        #color_image = np.asarray(color_frame.get_data(), np.uint8)
        #infrared_image = np.asarray(infrared_frame.get_data(), np.uint8)
        #depth_image = np.asarray(depth_frame.get_data(), np.uint16)
        
        #color_image = np.asanyarray(color_frame.get_data())
        #infrared_image = np.asanyarray(infrared_frame.get_data())
        #depth_image = np.asanyarray(depth_frame.get_data())
        
        #plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)) # frame is RGB
        #plt.imshow(color_image)
        #plt.show()
        #cv2.waitKey(0)
        
        #print("Color Image:")
        #print(type(color_image))
        #print(color_image.shape)
        #print(color_image.dtype)
        #print("IR Image:")
        #print(type(infrared_image))
        #print(infrared_image.shape)
        #print(infrared_image.dtype)
        #print("Depth Image: ")
        #print(type(depth_image))
        #print(depth_image.shape)
        #print(depth_image.dtype)
        
        #Success = False
pipeline.stop()

time_end = cv2.getTickCount()
time = (time_end - time_loading_end) / cv2.getTickFrequency()
time_loading = (time_loading_end - time_start) / cv2.getTickFrequency()

print("Found " + str(Frames) + " frames in " + str(time) + " seconds.")
print(str(time_loading) + " seconds where needed to load the .bag file")

exit()