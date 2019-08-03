#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
    int HoleFill = 1;
    int Magnitude = 2;
    bool bShowNonvisualizedDepthImages = false;
    if (argc >= 5)
    {
        std::string True = "TRUE";
        std::string input = argv[4];
        // convert input to upper
        for (auto & c : input) c = std::toupper(c);
        bShowNonvisualizedDepthImages = True.compare(input) == 0 ? true : false;
    }
    if (argc >= 4)
    {
        HoleFill = atoi(argv[3]);
    }
    if (argc >= 3)
    {
        Magnitude = atoi(argv[2]);
        if (Magnitude < 2)
        {
            Magnitude = 2;
        }
        if (Magnitude > 8)
        {
            Magnitude = 8;
        }
    }
    
    bool bPausePlayback = false;
    // define post-processing filters
    rs2::decimation_filter decimation_filter;
    rs2::hole_filling_filter hole_filling_filter;
    // configure filter parameter
    decimation_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, Magnitude);
    hole_filling_filter.set_option(RS2_OPTION_HOLES_FILL, HoleFill);
    
    rs2::colorizer color_map;
    rs2::frameset frames;
    rs2::frame depth;
    
    auto pipe = std::make_shared<rs2::pipeline>();
    
    //pipe->start();
    
    //rs2::device device = pipe->get_active_profile().get_device();
    
    //rs2::playback playback = device.as<rs2::playback>();
    //pipe->stop();
    pipe = std::make_shared<rs2::pipeline>();
    rs2::config cfg;
    cfg.enable_device_from_file(argv[1]);
    pipe->start(cfg);
    rs2::device device = pipe->get_active_profile().get_device();
    rs2::playback playback = device.as<rs2::playback>();
    
    cv::namedWindow("Visualized Depth Image", CV_WINDOW_AUTOSIZE);
    if (bShowNonvisualizedDepthImages)
    {
        cv::namedWindow("Non visualized Depth Frame", CV_WINDOW_AUTOSIZE);
    }
    //cv::namedWindow("Test1", CV_WINDOW_AUTOSIZE);
    
    if (bShowNonvisualizedDepthImages)
    {
        cv::namedWindow("Filtered non visualized Depth Frame", CV_WINDOW_AUTOSIZE);
    }
    cv::namedWindow("Filtered visualized Depth Image", CV_WINDOW_AUTOSIZE);
    
    while(true)
    {
        if (pipe->poll_for_frames(&frames))
        {
            rs2::depth_frame depth_frame = frames.get_depth_frame();
            rs2::depth_frame depth_frame_post_processing = depth_frame;
            //rs2::frame filtered_frame = decimation_filter.process(depth_frame_post_processing);
            //rs2::frame filtered_frame = hole_filling_filter.process(depth_frame_post_processing/*filtered_frame*/);
            rs2::frame filtered_frame = depth_frame_post_processing.as<rs2::frame>();
            
            //depth = color_map.process(frames.get_depth_frame());
            rs2::frame depth_filtered = color_map.process(filtered_frame);
            depth = color_map.process(depth_frame);
            const int DepthWidth = depth.as<rs2::video_frame>().get_width();
            const int DepthHeight = depth.as<rs2::video_frame>().get_height();
            
            const int FilteredWidth = depth_filtered.as<rs2::video_frame>().get_width();
            const int FilteredHeight = depth_filtered.as<rs2::video_frame>().get_height();
            
            cv::Mat DepthImageFiltered(cv::Size(FilteredWidth, FilteredHeight), CV_8UC3, (void*)depth_filtered.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat NonVisDepthImageFiltered(cv::Size(FilteredWidth, FilteredHeight), CV_16U, (void*)filtered_frame.get_data(), cv::Mat::AUTO_STEP);
            
            //cv::resize(DepthImageFiltered, DepthImageFiltered, cv::Size(640, 480), 0, 0, CV_INTER_LINEAR);
            //cv::resize(NonVisDepthImageFiltered, NonVisDepthImageFiltered, cv::Size(640, 480), 0, 0, CV_INTER_LINEAR);
            cv::Mat BilateralFilteredImage;
            cv::bilateralFilter(DepthImageFiltered, BilateralFilteredImage, 7, 1000, 1000);
            
            cv::Mat DepthImage(cv::Size(DepthWidth, DepthHeight), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat NonVisDepthImage(cv::Size(DepthWidth, DepthHeight), CV_16U, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
            
            //cv::Mat TestMat;
            //cv::cvtColor(NonVisDepthImage, TestMat, );
            
            if (bShowNonvisualizedDepthImages)
            {
                cv::imshow("Non visualized Depth Frame", NonVisDepthImage);
            }
            cv::imshow("Visualized Depth Image", DepthImage);
            //cv::imshow("Test1", TestMat);
            
            if (bShowNonvisualizedDepthImages)
            {
                cv::imshow("Filtered non visualized Depth Frame", NonVisDepthImageFiltered);
            }
            cv::imshow("Filtered visualized Depth Image", /*DepthImageFiltered*/BilateralFilteredImage);
            
        }
        int KeyPressed = cv::waitKey(1);
        if (KeyPressed == 32)
        {
            if (bPausePlayback)
            {
                playback.resume();
                bPausePlayback = false;
            }
            else
            {
                playback.pause();
                bPausePlayback = true;
            }
        }
        else if (KeyPressed >= 0)
        {
            break;
        }
    }
    
    
    
    return 0;
}
