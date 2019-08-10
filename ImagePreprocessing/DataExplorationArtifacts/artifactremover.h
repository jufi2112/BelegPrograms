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
     */
    void RecursiveMedianFilter(const cv::Mat& Src, cv::Mat& Dst, const int KernelRadius);
};

#endif // ARTIFACTREMOVER_H