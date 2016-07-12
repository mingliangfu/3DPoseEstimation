#pragma once

#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <iomanip>

using namespace Eigen;
using namespace std;
using namespace cv;

namespace sz { // For Wadim

using Instance = pair<string,Isometry3f>;
struct Frame
{
    int nr;
    Mat color, depth, cloud, mask, hues, normals;
    vector<Instance, Eigen::aligned_allocator<Instance>> gt;
};

struct Sample
{
    Mat data, label;
    void copySample(Sample sample)
    {
        sample.data.copyTo(this->data);
        this->label = sample.label;
    }
};

struct Benchmark
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vector<Frame> frames;
    vector< pair<string,string> > models;
    Matrix3f cam;
};

struct Background
{
    Mat color, depth, normals;
};

struct Triplet
{
    Sample anchor, puller, pusher0, pusher1, pusher2;
};

inline void loadbar( string label, unsigned int x, unsigned int n, unsigned int w = 20)
{
    if ( (x != n) && (x % (n/100+1) != 0) ) return;

    float ratio  =  x/(float)n;
    unsigned int   c      =  ratio * w;

    clog << label << setw(3) << (int)(ratio*100) << "% [";
    for (size_t x=0; x<c; x++) clog << "=";
    for (size_t x=c; x<w; x++) clog << " ";
    clog << "]\r" << flush;

    if ( x == n ) clog << endl;
}

inline Mat showRGBDPatch(const Mat &patch, bool show/*=true*/)
{
    vector<Mat> channels;
    cv::split(patch,channels);

    Mat RGB,D,NOR,out;
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);

    if (channels.size()==4)
    {
        out = Mat(patch.rows,patch.cols*2,CV_32FC3);
        RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));
        D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));
    }
    else if (channels.size()==7)
    {
        out = Mat(patch.rows,patch.cols*3,CV_32FC3);
        cv::merge(vector<Mat>({abs(channels[4]),abs(channels[5]),abs(channels[6])}),NOR);
        RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));
        D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));
        NOR.copyTo(out(Rect(patch.cols*2,0,patch.cols,patch.rows)));
    }

    if (show){imshow("Patch ", out); waitKey();}
    return out;
}

template<typename T>
inline std::vector<T> to_array(const std::string& s)
{
  std::vector<T> result;
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, ',')) result.push_back(boost::lexical_cast<T>(item));
  return result;
}

}

