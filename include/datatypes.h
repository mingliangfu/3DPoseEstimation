#ifndef DATATYPES_H
#define DATATYPES_H

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <iomanip>


using namespace Eigen;
using namespace std;
using namespace cv;

struct Frame
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int nr;
    Mat color, depth, cloud, mask, normals;
    Isometry3f gt;
};

struct Benchmark
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vector<Frame, Eigen::aligned_allocator<Frame> > frames;
    Matrix3f cam;
};

struct Sample
{
    Mat data, label;
};

struct Triplet
{
    Sample anchor, puller, pusher;
};

struct TripletWang
{
    Sample anchor, puller, pusher0, pusher1, pusher2;
};

struct Pair
{
    Sample anchor, puller;
};

struct TripletsPairs
{
    vector<Triplet> triplets;
    vector<Pair> pairs;
};

static inline void loadbar( string label, unsigned int x, unsigned int n, unsigned int w = 20)
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

inline Mat showRGBDPatch(const Mat &patch, bool show=true)
{
    vector<Mat> channels;
    //cv::split((patch+1.f)*0.5f,channels);
    cv::split(patch,channels);

    Mat RGB,D,out(patch.rows,patch.cols*2,CV_32FC3);

    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));

    cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);
    D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));

    if(show) {imshow("R G B D",out); waitKey();}
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


#endif // DATATYPES_H
