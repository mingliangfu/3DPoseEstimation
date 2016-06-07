#ifndef DATATYPES_H
#define DATATYPES_H

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>

#include <iostream>
#include <iomanip>


using namespace Eigen;
using namespace std;
using namespace cv;

struct Frame
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int nr;
    Mat color, depth, cloud, mask;
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




#endif // DATATYPES_H
