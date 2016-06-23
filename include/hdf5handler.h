#ifndef HDF5HANDLER_H
#define HDF5HANDLER_H

#include <iostream>
#include <H5Cpp.h>
#include "datatypes.h"

using namespace std;

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

#endif // HDF5HANDLER_H
