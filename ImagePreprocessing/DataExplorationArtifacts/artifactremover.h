#ifndef ARTIFACTREMOVER_H
#define ARTIFACTREMOVER_H

namespace cv
{
    class Mat;
}

class ArtifactRemover
{
public:
    ArtifactRemover();
    
    /**
     * @brief Removes artifacts by recursively applying a median filter that operates on non-artifact pixels only. This method got proposed by Lai et al. in 'A Large-Scale Hierarchical Multi-View RGB-D Object Dataset'
     * @param Src The original 16-bit depth map
     * @param Dst The filtered depth map with no remaining artifacts
     * @param KernelRadius The median kernel's radius
     * @return True of the function succeded, false otherwise
     */
    bool RecursiveMedianFilter(const cv::Mat& Src, cv::Mat& Dst, const int KernelRadius);
    
    /**
     * @brief Performs convolution on the src image to fill holes. Convolution is applied as described by Schmeing and Jiang in 'Color Segmentation Based Depth Image Filtering' which is based on Knutsson and Westin's 'Normalized and Differential Convolution'
     * @param Src The original 16-bit depth map
     * @param Dst The filtered depth map
     * @param KernelRadius The maximum distance from the center pixel at which pixels should be considered neighbors and get selected
     * @param Sigma Sigma of gaussian function
     * @return True if the function succeded, false otherwise
     */
    bool NormalizedConvolution(const cv::Mat& Src, cv::Mat* Dst, const int KernelRadius, const double Sigma);
    
private:
    
    /**
     * @brief 2D gaussian for given center point and other point
     * @param CenterX X coordinate of the center point
     * @param CenterY Y coordinate of the center point
     * @param Px X coordinate of the other point
     * @param Py Y coordinate of the other point
     * @param SigmaX Spread of X
     * @param SigmaY Spread of Y
     * @return
     */
    double Gaussian(const int CenterX, const int CenterY, const int Px, const int Py, const double SigmaX, const double SigmaY);
};

#endif // ARTIFACTREMOVER_H