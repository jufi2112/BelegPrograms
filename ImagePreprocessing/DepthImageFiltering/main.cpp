/* Created 14/10/2019 by Julien Fischer */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <experimental/filesystem>
#include <vector>
#include <string>

cv::Mat ApplyFilteringOperation(const cv::Mat& DepthImage);

int main(int argc, char** argv)
{
    if (argc <= 1)
    {
        std::cout << "Please specify the directory that contains the subfolders with the depth images.\n";
        return -1;
    }
    std::string Path = argv[1];
    
    // png compression level
    std::vector<int> PNGCompression;
    PNGCompression.push_back(CV_IMWRITE_PNG_COMPRESSION);
    PNGCompression.push_back(0);
    
    // start time
    double time_start = cv::getTickCount();
    
    int FrameCount = 0;

    /* get all folders that contain the different recordings
     * Examples: 'Night', 'Day', 'Cloudy' */
    std::vector<std::string> FoldersRecording;
    for (auto& p : std::experimental::filesystem::directory_iterator(Path))
    {
        // see@ https://en.cppreference.com/w/cpp/experimental/fs/file_type
        if (p.status().type() == std::experimental::filesystem::file_type::directory)
        {
            FoldersRecording.push_back(p.path().string());
        }
    }
    
    /* for every recording, get all scene folders
     * Examples: 'Scene_1', 'Scene_2', 'Scene_3'... */  
    for (const std::string& Recording : FoldersRecording)
    {
        std::vector<std::experimental::filesystem::path> FoldersScene;
        for (auto& p : std::experimental::filesystem::directory_iterator(Recording))
        {
            if (p.status().type() == std::experimental::filesystem::file_type::directory)
            {
                FoldersScene.push_back(p.path());
            }
        }
        
        /* for every scene, go into the 'Depth_Raw' folder and get all .png files from there */
        for (std::experimental::filesystem::path& Scene : FoldersScene)
        {
            // create directory 'Depth_Processed'
            std::experimental::filesystem::path Depth_Processed = Scene / "Depth_Processed";
            std::experimental::filesystem::create_directory(Depth_Processed);
            
            std::string SceneBasePath = Scene.string();
            Scene /= "Depth_Raw";
            if (!std::experimental::filesystem::exists(Scene))
            {
                // 'Depth_Raw' doesn't exist, ignore this scene folder
                std::cout << "Folder 'Depth_Raw' doesn't exist in directory " << Scene.string() << " Ignoring Scene\n";
                continue;
            }
            for (auto& p : std::experimental::filesystem::directory_iterator(Scene))
            {
                if (p.status().type() == std::experimental::filesystem::file_type::regular)
                {
                    // check if this is a .png file
                    if (p.path().extension() == ".png")
                    {
                        // load with opencv
                        cv::Mat DepthImage = cv::imread(p.path().string(), cv::IMREAD_ANYDEPTH);
                        cv::imwrite((Depth_Processed / p.path().filename()).string(), ApplyFilteringOperation(DepthImage), PNGCompression);
                        FrameCount++;
                    }
                }
            }         
        }
    }
    
    // end time
    double time_end = cv::getTickCount();
    double time = (time_end - time_start) / cv::getTickFrequency();
    
    std::cout << "Finished processing all " << FrameCount << " frames after " << time << " seconds.\n";
    
    return 0; 
}

cv::Mat ApplyFilteringOperation(const cv::Mat& DepthImage)
{
    cv::Size Size = DepthImage.size();
    // create intermediate matrix with same size
    cv::Mat Filtered(Size, CV_16U);
    // apply the filtering operation
    cv::medianBlur(DepthImage, Filtered, 7);
    return Filtered;
}
