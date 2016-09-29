#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <boost/filesystem.hpp>

#include <FastNoise.h>

#include "datatypes.h"
#include "helper.h"

using namespace std;
using namespace cv;
using namespace boost;

namespace sz {

class bgfill
{
public:
    bgfill();
    void loadBackgrounds(string backgrounds_path, int count=-1);
    void randomRealFill(Mat &patch);
    void randomColorFill(Mat &patch);
    void randomShapeFill(Mat &patch);
    void randomFractalFill(Mat &patch);

private:
    std::random_device ran;
    vector<Background> backgrounds;

};

}
