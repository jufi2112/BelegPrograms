#ifndef CANVAS_H
#define CANVAS_H
#include <vector>

namespace cv
{
    class Mat;
}

class Canvas
{
public:
    Canvas();
    
    cv::Mat MakeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);
    
    
};

#endif // CANVAS_H