#include <iostream>
#include <vector>
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "canvas.h"

/* align depth to color stream and display both streams */
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: <path to .bag> [Align to depth]" << std::endl;
        return -1;
    }
    
    std::string Path = argv[1];
    
    bool bAlignToDepth = false;
    bool bShouldPlay = true;
    
    if (argc >= 3)
    {
        std::string AlignOption = argv[2];
        if (AlignOption == "no" || AlignOption == "No" || AlignOption == "NO")
        {
            bAlignToDepth = false;
        }
        else
        {
            bAlignToDepth = true;
        }
    }
    
    rs2::colorizer ColorMap;
    ColorMap.set_option(RS2_OPTION_COLOR_SCHEME, 0);
    rs2::align Align(RS2_STREAM_COLOR);
    rs2::align AlignDepth(RS2_STREAM_DEPTH);
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_device_from_file(Path);
    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::device device = profile.get_device();
    rs2::playback playback = device.as<rs2::playback>();
    playback.set_real_time(false);
    
    rs2::frameset Frames;
    rs2::frameset FramesAligned;
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    
    while(true)
    {
        if (bShouldPlay)
        {
            if (pipe.poll_for_frames(&Frames))
            {
                rs2::depth_frame DepthFrame = Frames.get_depth_frame();
                rs2::video_frame DepthFrameVisualized = ColorMap.process(DepthFrame);
                if (bAlignToDepth)
                {
                    FramesAligned = AlignDepth.process(Frames);
                }
                else
                {
                    FramesAligned = Align.process(Frames);
                }
                    
                rs2::video_frame VideoFrame = FramesAligned.first(RS2_STREAM_COLOR);
                rs2::depth_frame AlignedDepthFrame = FramesAligned.get_depth_frame();
                rs2::video_frame AlignedDepthFrameVisualized = ColorMap.process(AlignedDepthFrame);
                
                if (!VideoFrame || !AlignedDepthFrame)
                {
                    continue;
                }
                
                const int ColorWidth = VideoFrame.get_width();
                const int ColorHeight = VideoFrame.get_height();
                
                const int AlignedDepthWidth = AlignedDepthFrame.get_width();
                const int AlignedDepthHeight = AlignedDepthFrame.get_height();
                
                const int DepthWidth = DepthFrame.get_width();
                const int DepthHeight = DepthFrame.get_height();
                
                cv::Mat VideoImage(cv::Size(ColorWidth, ColorHeight), CV_8UC3, (void*)VideoFrame.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat AlignedDepthImage(cv::Size(AlignedDepthWidth, AlignedDepthHeight), CV_16U, (void*)AlignedDepthFrame.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat AlignedDepthImageVisualized(cv::Size(AlignedDepthWidth, AlignedDepthHeight), CV_8UC3, (void*)AlignedDepthFrameVisualized.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat DepthImage(cv::Size(DepthWidth, DepthHeight), CV_16U, (void*)DepthFrame.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat DepthImageVisualized(cv::Size(DepthWidth, DepthHeight), CV_8UC3, (void*)DepthFrameVisualized.get_data(), cv::Mat::AUTO_STEP);
                
                std::vector<cv::Mat> Images {DepthImage, DepthImageVisualized, VideoImage, AlignedDepthImage, AlignedDepthImageVisualized, VideoImage};
                Canvas Canvas;
                cv::Mat I = Canvas.MakeCanvas(Images, ColorHeight * 2, 2);
                cv::imshow("Image", I);
                
            }
        }
        const int KeyPressed = cv::waitKey(1);
        if (KeyPressed == 32)
        {
            bShouldPlay = !bShouldPlay;
        }
        else if (KeyPressed >= 0)
        {
            playback.stop();
            pipe.stop();
            return 0;
        }
    }
    
    return 0;
}
