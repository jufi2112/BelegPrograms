#include <iostream>
#include <string>
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// basic code taken from: https://github.com/IntelRealSense/librealsense/tree/development/examples/capture

// image resolutions
#define COLOR_IMAGE_WIDTH 640
#define COLOR_IMAGE_HEIGHT 480
#define DEPTH_IMAGE_WIDTH 640
#define DEPTH_IMAGE_HEIGHT 480
#define IR_IMAGE_WIDTH 640
#define IR_IMAGE_HEIGHT 480


/**
* TODO:
* 
*   * only sample each n-th frame (because two adjacent frames will not provide many new information?)
*   * if frame rate is visible in stored cv mats, remove frame filter
*   * should the depth image be colored or uncolored?
*   * map rgb stream to depth stream (see https://github.com/IntelRealSense/librealsense/tree/master/examples/align)
* 
*   * SR300 stream do only work on after usb 3.0 port when power is connected
*/
int main(int argc, char** argv)
{
    std::string FilePath = "";
    bool bSaveImages = false;
    int FrameCount = 0;   
    // process command line arguments
    
    if (argc < 2 || argc > 3)
    {
        std::cout << "Invalid command line arguments. Usage: <frame save directory> <0 - don't save frames, 1 - save frames>" << std::endl;
        return -1;
    }
    
    FilePath = argv[1];
    if (atoi(argv[2]) == 0)         // treats all invalid input as no
    {
        bSaveImages = false;
    }
    else
    {
        bSaveImages = true;
    }
    
    // controls that image recording only starts after a key is pressed
    bool bRecordingStarted = false;
    
    std::cout << "Save directory choosen: " << FilePath << std::endl;
    std::cout << "Save images?: " << (bSaveImages ? "yes" : "no") << std::endl;
    
    // create OpenCV windows
    cv::namedWindow("Depth Stream", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Color Stream", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("IR Stream", CV_WINDOW_AUTOSIZE);    
    
    // color visualization of depth data
    rs2::colorizer color_map;
    //color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2);
    
    // used to show stream rates for every stream
    rs2::rates_printer printer;
    
    rs2::pipeline pipe;
    // default stream configuration is depth and color
    
    // create streaming configuration with depth, color and ir stream
    rs2::config StreamingConfig;
    StreamingConfig.enable_stream(RS2_STREAM_DEPTH, -1, DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT, RS2_FORMAT_Z16, 30);     // uncolored will be RS2_FORMAT_Z16, see https://github.com/IntelRealSense/librealsense/blob/master/doc/stepbystep/getting_started_with_openCV.md#displaying-infrared-frame
    StreamingConfig.enable_stream(RS2_STREAM_COLOR, -1, COLOR_IMAGE_WIDTH, COLOR_IMAGE_HEIGHT, RS2_FORMAT_BGR8, 30);
    StreamingConfig.enable_stream(RS2_STREAM_INFRARED, -1, IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT, RS2_FORMAT_Y8, 30);
    // check for errors in the configuration
    rs2::pipeline_profile profile;
    try
    {
        profile = StreamingConfig.resolve(pipe);
    }
    catch (const rs2::error &e)
    {
        std::cout << "ERROR: Could not resolve given streaming configuration." << std::endl << "Error is: '" << e.what() << "' in function '" << e.get_failed_function() << "'" << std::endl;
        return -1;
    }
    
    
    // print the streaming profiles
    std::vector<rs2::stream_profile> StreamingProfiles = profile.get_streams();
    for (rs2::stream_profile& stream : StreamingProfiles)
    {
        std::cout << "StreamingProfile used: " << stream.stream_type() << std::endl;
    }
    
    pipe.start(StreamingConfig);
    
    // drop the first 30 frames for auto-exposure stabilization
    rs2::frameset FramesToDrop;
    for (int i = 0; i < 30; ++i)
    {
        FramesToDrop = pipe.wait_for_frames();
    }
    
    while (true)
    {
        rs2::frameset data = pipe.wait_for_frames().              // wait for next set of synchronized frames from the camera
                apply_filter(printer);                            // print stream frame rates
                //apply_filter(color_map);                        // find and colorize depth data
    
        // get depth frame        
        rs2::depth_frame depth_raw = data.get_depth_frame();        // raw depth image (16 bit)
        rs2::frame depth_vis = color_map.process(depth_raw);        // depth image visualized (jet)
        cv::Mat DepthImage_Vis(cv::Size(DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT), CV_8UC3, (void*)depth_vis.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat DepthImage_Raw(cv::Size(DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT), CV_16U, (void*)depth_raw.get_data(), cv::Mat::AUTO_STEP);
                
        // get color frame
        rs2::video_frame color = data.get_color_frame();
        cv::Mat ColorImage(cv::Size(COLOR_IMAGE_WIDTH, COLOR_IMAGE_HEIGHT), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
    
        // get IR frame
        rs2::video_frame ir = data.get_infrared_frame();
        cv::Mat IRImage(cv::Size(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT), CV_8UC1, (void*)ir.get_data(), cv::Mat::AUTO_STEP);
        
        // show frames
        cv::imshow("Depth Stream", DepthImage_Vis);
        cv::imshow("Color Stream", ColorImage);
        cv::imshow("IR Stream", IRImage);
    
        if (bSaveImages && bRecordingStarted)
        {
            
            cv::imwrite(FilePath + "/Depth/" + std::to_string(FrameCount) + ".jpg", DepthImage_Vis);
            cv::imwrite(FilePath + "/DepthRaw/" + std::to_string(FrameCount) + ".png", DepthImage_Raw);
            cv::imwrite(FilePath + "/Color/" + std::to_string(FrameCount) + ".jpg", ColorImage);
            cv::imwrite(FilePath + "/IR/" + std::to_string(FrameCount) + ".jpg", IRImage);
            FrameCount++;
        }
   
        if (cv::waitKey(1) >= 0)
        {
            if (!bRecordingStarted)
            {
                bRecordingStarted = true;
                std::cout << "Recording started..." << std::endl;
            }
            else
            {
                std::cout << "Recording finished." << std::endl;
                break;
            }
        }
    }
    std::cout << "Exiting..." << std::endl;
        
    return 0;
}
