#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stack>

#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/lexical_cast.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;

namespace sz {

// Show progress bar
inline void loadbar(string label, unsigned int x, unsigned int n, unsigned int w = 20)
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

// Convert string to array of strings separated by ,
template<typename T>
inline std::vector<T> to_array(const std::string& s)
{
    std::vector<T> result;
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, ',')) result.push_back(boost::lexical_cast<T>(item));
    return result;
}

// Check if file exists
inline bool fexists(const string filename) {
    std::ifstream ifile(filename);
    return (bool)ifile;
}

// Show RGB-D patch
Mat showRGBDPatch(const Mat &patch, bool show = true);

Mat showTriplet(const Mat &p0,const Mat &p1,const Mat &p2,const Mat &p3,const Mat &p4, bool show = true);

// Compute normals
void depth2normals(const Mat &depth, Mat &normals, float fx, float fy, float ox, float oy);
inline void depth2normals(const Mat &depth, Mat &normals, Matrix3f &cam)
{depth2normals(depth,normals,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}

// Depth -> Cloud
void     depth2cloud(const Mat &depth, Mat &cloud, float fx, float fy, float ox, float oy);
inline Vector3f depth2cloud(Point p, float d, float fx, float fy, float ox, float oy)
{return Vector3f(d*(p.x-ox)/fx,d*(p.y-oy)/fy,d);}

inline void depth2cloud(const Mat &depth, Mat &cloud, Matrix3f &cam)
{depth2cloud(depth,cloud,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}

inline Vector3f depth2cloud(Point p, float d, Matrix3f &cam)
{return depth2cloud(p,d,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}

// Region growing
Mat growForeground(Mat &depth);

}
