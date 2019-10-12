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

def process_images(files, path, save_path_main, downsample_filter_magnitude=1, skip_existing_scenes=False, verbose=0):
    # configure pipeline
    #try:
    
    time_start_process = cv2.getTickCount()
    
    main_path_separated = os.path.split(path)
    main_path_to_scenes = os.path.join(save_path_main, main_path_separated[1])
    if verbose > 0:
        print("Main path to scenes is: " + main_path_to_scenes)
    
    Frames_overall = 0
    
    pipeline = rs.pipeline()
    config = rs.config()
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
    for file in tqdm(files):
        # skip file if it is marked as defect
        if file.split("_")[0] == "DEFECT":
            if verbose > 0:
                print("Found defect .bag file. Skipping...")
            continue
        
        time_start_one = cv2.getTickCount()
        
        file_split = file.split("_")
        
        save_path = main_path_to_scenes
        
        # this is the base folder for where to save the images from this .bag file to
        save_path = os.path.join(save_path, file_split[0] + "_" + file_split[1])
        
        # skip folder if we dont want to append to existing folders and the folder exists
        if skip_existing_scenes:
            if os.path.isdir(save_path):
                if verbose > 0:
                    print("Skipping existing folder: " + save_path)
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
        
        # initialize decimation filter -> not used anymore
        #decimation = rs.decimation_filter()
        #decimation.set_option(rs.option.filter_magnitude, downsample_filter_magnitude)
        
        if verbose > 0:
            print("Depth scale for " + file +  " is: " + str(depth_scale))
        
        # create folder structure to save extracted images to
        
        # create subfolder if not existant
        os.makedirs(save_path, exist_ok=True)
                
        # folder to save color images
        save_color = os.path.join(save_path, "Color")
        # folder to save infrared images
        save_ir = os.path.join(save_path, "Infrared")
        # folder to save raw depth images
        save_depth_raw = os.path.join(save_path, "Depth_Raw")
        # folder to save processed depth images
        #save_depth_processed = os.path.join(save_path, "Depth_Processed")
        
        # create these directories
        os.makedirs(save_color, exist_ok=True)
        os.makedirs(save_ir, exist_ok=True)
        os.makedirs(save_depth_raw, exist_ok=True)
        #os.makedirs(save_depth_processed, exist_ok=True)
        
        # create .txt file that contains the depth scale value
        with open(os.path.join(save_path, "Scale.txt"), 'w') as scale_file:
            scale_file.write(str(depth_scale))
            
        Success = True
        # check if there are already images in the subfolders (i.e. the current scene .bag is a _Part_2)
        # set Frames to the number of existing images (since naming is 1-based but Frames gets incremented at the start of the loop)
        existing_jpg = os.listdir(save_color)
        Frames = len([jpg for jpg in existing_jpg if Path(jpg).suffix == ".jpg"])
        if verbose > 0:
            print("For " + file + " " + str(Frames) + " frames are already present. Starting with index " + str(Frames+1))
        
        Frames_current_bag = 0
        
        while Success:
            Success, frames = pipeline.try_wait_for_frames()
                #print(Success)
            if Success is True:
                Frames += 1
                Frames_overall += 1
                Frames_current_bag += 1
                    #print(Frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame()
                
                # tell numpy explicitly what datatype we want (not necessary)
                
                color_image = np.asarray(color_frame.get_data(), dtype=np.uint8)
                ir_image = np.asarray(ir_frame.get_data(), dtype=np.uint16)
                depth_image_raw = np.asarray(depth_frame.get_data(), dtype=np.uint16)
                
                #depth_frame_proc = decimation.process(depth_frame)      
                #depth_image_proc = np.asarray(depth_frame_proc.get_data(), dtype=np.uint16)
                
                # convert all color images to BGR so they get adequately saved as RGB / infrared and depth image only have one channel, so they don't matter (imwrite assumes bgr images and saves them as rgb images)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # multiply depth image's values by the cameras scaling factor to retrieve absolut depth values in meter --> do this when reading the values in network
                # float64 values cannot be saved as images in opencv
                #depth_image_raw = np.asarray(depth_scale * depth_image_raw, dtype=np.float64)
                #depth_image_proc = np.asarray(depth_scale * depth_image_proc, dtype=np.float64)
                
                # save images to respective folders
                color_filepath = os.path.join(save_color, str(Frames) + ".jpg")
                ir_filepath = os.path.join(save_ir, str(Frames) + ".png")
                depth_raw_filepath = os.path.join(save_depth_raw, str(Frames) + ".png")
                #depth_processed_filepath = os.path.join(save_depth_processed, str(Frames) + ".png")
                
                cv2.imwrite(color_filepath, color_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                cv2.imwrite(ir_filepath, ir_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(depth_raw_filepath, depth_image_raw, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                #cv2.imwrite(depth_processed_filepath, depth_image_proc, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                                      
                
                #plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                #plt.imshow(ir_image, cmap="gray")
                #plt.imshow(depth_image, cmap="gray")
                #plt.pause(.1)
                #plt.draw()
        pipeline.stop()
        time_end_one = cv2.getTickCount()
        if verbose > 0:
            print("For " + file + " found " + str(Frames_current_bag) + " frames")

        if verbose > 1:
            time_one = (time_end_one - time_start_one) / cv2.getTickFrequency()
            print("Needed " + str(time_one) + " seconds to extract all images from " + file)
    
    time_end_process = cv2.getTickCount()
    if verbose > 1:
        time_process = (time_end_process - time_start_process) / cv2.getTickFrequency()
        print("Needed " + str(time_process) + " seconds to process " + main_path_separated[1])
    return Frames_overall               
                
    #finally:
        #pass
        

# object for parsing command line
Parser = argparse.ArgumentParser(description="Reads all recorded bag files from the provided folder and saves the included images to the provided output folder.")
Parser.add_argument("-i", "--input", type=str, help="Path to main folder that contains subfolders 'Outdoor_Lighting' and 'Indoor_Lighting' which contain the recorded bag files")
Parser.add_argument("-o", "--output", type=str, help="Path where the included images should be saved to")
Parser.add_argument("-d", "--decimation", type=int, default=1 ,help="Decimation filter magnitude. Values of 2 and 3 perform median downsampling, values greater than 3 perform mean downsampling")
Parser.add_argument("-s", "--skip", type=bool, default=False, help="Should scenes that already exist in the output location be skipped. Default is False. If activated, _Part_2.bag files will not be considered. This can also be used to only process .bag files that are not yet processed")
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
    
if not args.decimation:
    print("No decimation filter magnitude input. Assuming 1 (no downsampling")
    args.decimation = 1 # necessary with default=1 ?
    
time_start_overall = cv2.getTickCount()

# paths to subfolders
day_path = os.path.join(args.input, "Day")
night_path = os.path.join(args.input, "Night")
cloudy_path = os.path.join(args.input, "Cloudy_Conditions")

# check if paths exist
if not os.path.isdir(day_path):
    print("Could not find the directory " + day_path)
    exit()
    
if not os.path.isdir(night_path):
    print("Could not find the directory " + night_path)
    exit()
    
if not os.path.isdir(cloudy_path):
    print("Could not find the directory " + cloudy_path)
    exit()

print("Searching for .bag files in " + day_path + ", " + night_path + " and " + cloudy_path)

# gather all .bag files
day_available_files = os.listdir(day_path)
night_available_files = os.listdir(night_path)
cloudy_available_files = os.listdir(cloudy_path)

day_files = [file for file in day_available_files if Path(file).suffix == ".bag"]
night_files = [file for file in night_available_files if Path(file).suffix == ".bag"]
cloudy_files = [file for file in cloudy_available_files if Path(file).suffix == ".bag"]

print("Found " + str(len(day_files)) + " daytime .bag files")
print("Found " + str(len(night_files)) + " nighttime .bag files")
print("Found " + str(len(cloudy_files)) + " cloudy weather .bag files")

number_frames = 0

print("Starting preprocessing for files in " + day_path + " ...")
number_frames += process_images(day_files, day_path, args.output, args.decimation, args.skip, args.verbose)

print("Starting preprocessing for files in " + night_path + " ...")
number_frames += process_images(night_files, night_path, args.output, args.decimation, args.skip, args.verbose)

print("Starting preprocessing for files in " + cloudy_path + " ...")
number_frames += process_images(cloudy_files, cloudy_path, args.output, args.decimation, args.skip, args.verbose)

time_end_overall = cv2.getTickCount()
time_overall = (time_end_overall - time_start_overall) / cv2.getTickFrequency()

print("Finished processing all " + str(number_frames) + " frames in " + str(time_overall) + " seconds.")


exit()