#include <iostream>
#include <vector>
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/**
 * code by berak from https://stackoverflow.com/questions/23195522/opencv-fastest-method-to-check-if-two-images-are-100-same-or-not
 */
bool equal(const cv::Mat & a, const cv::Mat & b)
{
    if ( (a.rows != b.rows) || (a.cols != b.cols) )
        return false;
    cv::Scalar s = cv::sum( a - b );
    return (s[0]==0) && (s[1]==0) && (s[2]==0);
}

/**
 * code taken from vinvinod at https://stackoverflow.com/questions/5089927/show-multiple-2-3-4-images-in-the-same-window-in-opencv
 * @brief makeCanvas Makes composite image from the given images
 * @param vecMat Vector of Images.
 * @param windowHeight The height of the new composite image to be formed.
 * @param nRows Number of rows of images. (Number of columns will be calculated
 *              depending on the value of total number of images).
 * @return new composite image.
 */
    cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
            int N = vecMat.size();
            nRows  = nRows > N ? N : nRows; 
            int edgeThickness = 10;
            int imagesPerRow = ceil(double(N) / nRows);
            int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
            int maxRowLength = 0;

            std::vector<int> resizeWidth;
            for (int i = 0; i < N;) {
                    int thisRowLen = 0;
                    for (int k = 0; k < imagesPerRow; k++) {
                            double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
                            int temp = int( ceil(resizeHeight * aspectRatio));
                            resizeWidth.push_back(temp);
                            thisRowLen += temp;
                            if (++i == N) break;
                    }
                    if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
                            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
                    }
            }
            int windowWidth = maxRowLength;
            cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

            for (int k = 0, i = 0; i < nRows; i++) {
                    int y = i * resizeHeight + (i + 1) * edgeThickness;
                    int x_end = edgeThickness;
                    for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
                            int x = x_end;
                            cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
                            cv::Size s = canvasImage(roi).size();
                            // change the number of channels to three
                            cv::Mat target_ROI(s, CV_8UC3);
                            if (vecMat[k].channels() != canvasImage.channels()) {
                                if (vecMat[k].channels() == 1) {
                                    cv::cvtColor(vecMat[k], target_ROI, CV_GRAY2BGR);
                                }
                            } else {             
                                vecMat[k].copyTo(target_ROI);
                            }
                            cv::resize(target_ROI, target_ROI, s);
                            if (target_ROI.type() != canvasImage.type()) {
                                target_ROI.convertTo(target_ROI, canvasImage.type());
                            }
                            target_ROI.copyTo(canvasImage(roi));
                            x_end += resizeWidth[k] + edgeThickness;
                    }
            }
            return canvasImage;
    }

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Too less arguments. Usage is: <filename> <filter option>" << std::endl;
        return 0;
    }
    std::string Path = argv[1];
    int OptionHoleFilling = 0;
    OptionHoleFilling = atoi(argv[2]);
    /** 
     * 0 - fill_from_left
     * 1 - farest_from_around
     * 2 - nearest_from_around
     */
    
    
    bool bPausePlayback = false;
    bool bShouldPlayback = true;
    
    // define postprocessing filters
    rs2::hole_filling_filter HoleFillingFilter;
    rs2::spatial_filter SpatialFilter;
    rs2::colorizer ColorMap;
    
    // configure filter parameters
    HoleFillingFilter.set_option(RS2_OPTION_HOLES_FILL, OptionHoleFilling);
    SpatialFilter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.55f);
    SpatialFilter.set_option(RS2_OPTION_HOLES_FILL, 3);
    SpatialFilter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    
    auto pipe = std::make_shared<rs2::pipeline>();
    rs2::config cfg;
    cfg.enable_device_from_file(Path);
    pipe->start(cfg);
    rs2::device device = pipe->get_active_profile().get_device();
    rs2::playback playback = device.as<rs2::playback>(); 
    playback.set_real_time(false);
    
    rs2::frameset Frames;
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    //cv::namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
    //cv::namedWindow("Filtered Depth Image", CV_WINDOW_AUTOSIZE);
    cv::Mat FirstImage;
    
    int FrameCount = 0;
    
    while(true)
    {
        if (pipe->poll_for_frames(&Frames) && bShouldPlayback)
        {
            FrameCount++;
            rs2::depth_frame DepthFrame = Frames.get_depth_frame();
            rs2::video_frame ColorFrame = Frames.get_color_frame();
            
            const int DepthWidth = DepthFrame.as<rs2::video_frame>().get_width();
            const int DepthHeight = DepthFrame.as<rs2::video_frame>().get_height();
            const int ColorWidth = ColorFrame.get_width();
            const int ColorHeight = ColorFrame.get_height();
            
            rs2::frame VisualizedDepthFrame = ColorMap.process(DepthFrame);
            //rs2::frame FilteredFrame = HoleFillingFilter.process(DepthFrame);
            rs2::frame FilteredFrame = SpatialFilter.process(DepthFrame);
            rs2::frame DoubleFilteredFrame = HoleFillingFilter.process(FilteredFrame);
            rs2::frame FilteredVisualizedDepthFrame = ColorMap.process(FilteredFrame);
            rs2::frame DoubleFilteredVisualizedDepthFrame = ColorMap.process(DoubleFilteredFrame);
            
            cv::Mat Image(cv::Size(ColorWidth, ColorHeight), CV_8UC3, (void*)ColorFrame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat VisualizedImage(cv::Size(DepthWidth, DepthHeight), CV_8UC3, (void*)VisualizedDepthFrame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat FilteredImage(cv::Size(DepthWidth, DepthHeight), CV_8UC3, (void*)FilteredVisualizedDepthFrame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat DoubleFilteredImage(cv::Size(DepthWidth, DepthHeight), CV_8UC3, (void*)DoubleFilteredVisualizedDepthFrame.get_data(), cv::Mat::AUTO_STEP);
            if (FrameCount == 1)
            {
                Image.copyTo(FirstImage);
            }
            
            if (FrameCount != 1 && equal(FirstImage, Image))
            {
                std::cout << "Last frame detected. Stopping playback..." << std::endl;
                FrameCount--;
                bShouldPlayback = false;
            }
            else
            {
                std::vector<cv::Mat> Images {Image, VisualizedImage, FilteredImage, DoubleFilteredImage};
                cv::Mat Canvas = makeCanvas(Images, 960, 2);
                cv::imshow("Image", Canvas);
                //cv::imshow("Depth Image", VisualizedImage);
            }
        }
        int KeyPressed = cv::waitKey(1);
        if (KeyPressed == 32)
        {
            if (!bPausePlayback)
            {
                playback.pause();
                bPausePlayback = !bPausePlayback;
            }
            else
            {
                playback.resume();
                bPausePlayback = !bPausePlayback;
            }
        }
        else if (KeyPressed >= 0)
        {
            break;
        }
    }
    
    std::cout << "#Frames: " << FrameCount << std::endl;
    return 0;
}
