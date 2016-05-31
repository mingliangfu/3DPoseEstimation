#ifndef NETWORKSOLVER_H
#define NETWORKSOLVER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include "model.h"
#include "datatypes.h"
#include "hdf5handler.h"

class networkSolver
{
public:
    networkSolver(string network_path, string hdf5_path);
    TripletsPairs buildTripletsPairs(vector<string> used_models);
    void setNetworkParameters();
    void trainNet(vector<string> used_models, string net_name, int resume_iter);
    Mat computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples);
    void testNet();
//    void testKNN(bool realData);
private:
    hdf5Handler h5;
    string network_path;
    string hdf5_path;
    std::random_device ran;
};

#endif // NETWORKSOLVER_H
