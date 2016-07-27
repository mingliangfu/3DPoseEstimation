#pragma once

#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

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
        sample.label.copyTo(this->label);
    }

    Quaternionf getQuat() const
    {
        Quaternionf quat;
        for (int j = 0; j < 4; ++j)
            quat.coeffs()(j) = label.at<float>(0,1+j);
        return quat;
    }

    Vector3f getTrans() const
    {
        Vector3f trans;
        for (size_t tr = 0; tr < 3; ++tr)
            trans[tr] = label.at<float>(0,5+tr);
        return trans;
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

}

