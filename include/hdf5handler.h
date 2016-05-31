#ifndef HDF5HANDLER_H
#define HDF5HANDLER_H

#include<iostream>
#include <H5Cpp.h>
#include "datatypes.h"

using namespace std;

class hdf5Handler
{
public:
    hdf5Handler();
    vector<Sample> read(string filename);
    void write(string filename, vector<Sample> &samples);
};

#endif // HDF5HANDLER_H
