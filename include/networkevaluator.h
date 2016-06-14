#ifndef NETWORKEVALUATOR_H
#define NETWORKEVALUATOR_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>
#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"

#include "datatypes.h"

using namespace std;
using namespace cv;
using namespace boost;

class networkEvaluator
{
public:
    networkEvaluator();
    static Mat computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples);
    static void computeKNNAccuracy(vector<vector<vector<int> > > &maxSimTmpl, vector<vector<vector<int> > > &maxSimKNNTmpl);
    static void visualizeManifold(caffe::Net<float> &CNN, const vector<vector<Sample> > &templates, int iter);
};

#endif // NETWORKEVALUATOR_H