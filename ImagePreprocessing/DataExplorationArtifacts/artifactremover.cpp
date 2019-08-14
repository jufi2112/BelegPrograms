#include "artifactremover.h"
#include <math.h>
#include <opencv2/core.hpp>
#include <iostream>

ArtifactRemover::ArtifactRemover()
{
    
}


bool ArtifactRemover::RecursiveMedianFilter(const cv::Mat& Src, cv::Mat& Dst, const int KernelRadius)
{
    if (Src.depth() != CV_16U)
    {
        std::cout << "Src image must be 16-bit grayscale!" << std::endl;
        return false;
    }
    
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
    return true;
    
}


bool ArtifactRemover::NormalizedConvolution(const cv::Mat& Src, cv::Mat* Dst, const int KernelRadius, const double Sigma)
{
    Src.copyTo(*Dst);
    // iterate through all pixels and only
    for (int Row = 0; Row < Src.rows; ++Row)
    {
        for (int Col = 0; Col < Src.cols; ++Col)
        {
            // "A hole pixel x is filled with a weighted sum of the depth values of its non-hole neighboring pixels[...]" Schmeing and Jiang
            if (Src.at<unsigned int>(Row,Col) == 0)
            {
                double SigmaNeighbors = 0;
                double SigmaGauss = 0;
                // gather all neighboring pixels
                for (int y = -KernelRadius; y <= KernelRadius; ++y)     // y is height axis of image (row)
                {
                    // skip if pixels would lie outside of image (above or below) (assume pixel value is 0)
                    if (((Row + y) < 0) || ((Row + y) > (Src.rows -1)))
                    {
                        continue;
                    }
                    for (int x = -KernelRadius; x <= KernelRadius; ++x)     // x is width axis of image (col)
                    {
                        // skip if pixel would lie outside of image (to the left or to the right) (assume pixel value is 0)
                        if (((Col + x) < 0) || ((Col + x) > (Src.cols -1)))
                        {
                            continue;
                        }
                        // only use pixel if its value is not zero
                        unsigned int NeighborValue = Src.at<unsigned int>(Row+y, Col+x);
                        if (NeighborValue > 0)
                        {
                            SigmaNeighbors += (NeighborValue * Gaussian(Col, Row, x, y, Sigma, Sigma));
                            SigmaGauss += Gaussian(Col, Row, x, y, Sigma, Sigma);                            
                        }
                    }
                }
                if (SigmaGauss == 0)
                {
                    SigmaGauss = 1;
                }
                unsigned int NewValue = (unsigned int)(SigmaNeighbors / SigmaGauss);
                Dst->at<unsigned int>(Row, Col) = NewValue;       // <-- error at this line
            }
        }
    }
//    std::cout << "test" << std::endl;
//    std::cout << "Input: " << Src.cols << "x" << Src.rows << "; Output: " << Dst.cols << "x" << Dst.rows << std::endl;
    return true;
}

double ArtifactRemover::Gaussian(const int CenterX, const int CenterY, const int Px, const int Py, const double SigmaX, const double SigmaY)
{
    double FirstExpression = std::pow(Px - CenterX, 2) / (2 * std::pow(SigmaX, 2));
    double SecondExpression = std::pow(Py - CenterY, 2) / (2 * std::pow(SigmaY, 2));
    
    return std::exp(- (FirstExpression + SecondExpression));
}
