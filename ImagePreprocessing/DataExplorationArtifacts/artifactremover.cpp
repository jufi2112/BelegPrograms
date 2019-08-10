#include "artifactremover.h"
#include <math.h>
#include <opencv2/core.hpp>
#include <iostream>

ArtifactRemover::ArtifactRemover()
{
    
}


void ArtifactRemover::RecursiveMedianFilter(const cv::Mat& Src, cv::Mat& Dst, const int KernelRadius)
{
    std::cout << "Starting recursive median filter" << std::endl;
    int Histogram[65536] = {};
    
    bool bArtifactsRemaining = false;
    
    cv::Mat Buffer;
    Src.copyTo(Buffer);
    
    for (int Row = 0; Row < Src.rows; ++Row)
    {
        if ((Row < KernelRadius) || ((Row + KernelRadius) > (Src.rows - 1)))
        {
            continue;
        }
        
        std::fill(Histogram, Histogram + 65536, 0);
        
        bool bIsFirstPixel = true;
        
        for (int Col = 0; Col < Src.cols; ++Col)
        {
            if ((Col < KernelRadius) || ((Col + KernelRadius) > (Src.cols - 1)))
            {
                continue;
            }
            
            if (bIsFirstPixel)
            {
                bIsFirstPixel = false;
                // fill histogram with whole kernel if first pixel in a row
                for (int X = -KernelRadius; X <= KernelRadius; ++X)
                {
                    for (int Y = -KernelRadius; Y <= KernelRadius; ++Y)
                    {
                        unsigned short DepthValue = Src.at<unsigned short>(Row + X, Col + Y);
                        Histogram[DepthValue]++;
                    }
                }
            }
            // only update histogram if not first pixel in a row
            else
            {
                for (int X = -KernelRadius; X <= KernelRadius; ++X)
                {
                    // remove pixels from leftmose col
                    unsigned short DepthValueToRemove = Src.at<unsigned short>(Row + X, Col - KernelRadius -1);
                    Histogram[DepthValueToRemove]--;
                    
                    // insert new col of pixels from right
                    unsigned short DepthValueToAdd = Src.at<unsigned short>(Row + X, Col + KernelRadius);
                    Histogram[DepthValueToAdd]++;
                }
            }
            
            /* pick median element from histogram if current pixel is an artifact*/
            if (Src.at<unsigned short>(Row, Col) == 0)
            {
                const int KernelSize = (int)std::pow(2 * KernelRadius +1, 2);
                const int RemainingSize = KernelSize - Histogram[0];
                
                if (RemainingSize == 0)
                {
                    // if an artifact remains in the image, the median filter is applied again
                    bArtifactsRemaining = true;
                }
                else
                {
                    int Number = 0;
                    const int HalfSize = (int) (RemainingSize / 2);
                    for (unsigned int i = 1; i < 65536; ++i)
                    {
                        Number += Histogram[i];
                        if (Number >= HalfSize)
                        {
                            Buffer.at<unsigned int>(Row, Col) = i;
                        }
                    }
                }   
            } 
        }
    }
    Buffer.copyTo(Dst);
    return;
    
}
