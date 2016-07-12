#pragma once

#include <iostream>
#include <H5Cpp.h>
#include "datatypes.h"

using namespace std;


namespace sz { // For Wadim

class hdf5Handler
{
public:
    hdf5Handler();
    vector<Sample> read(string filename);
    void write(string filename, vector<Sample> &samples);
    Isometry3f readBBPose(string filename);
    Matrix3f readBBIntristicMats(string filename);
    vector<Isometry3f,Eigen::aligned_allocator<Isometry3f>> readBBTrans(string filename);
    Mat readBBDepth(string filename);
};

}

