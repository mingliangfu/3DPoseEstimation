#pragma once

#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"

#include "datatypes.h"
#include "helper.h"
#include "datasetmanager.h"

using namespace std;
using namespace cv;
using namespace boost;

namespace sz {

class networkEvaluator
{
public:
    networkEvaluator();
    static Mat computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples);
    static void computeKNNAccuracy(vector<vector<vector<int> > > &maxSimTmpl, vector<vector<vector<int> > > &maxSimKNNTmpl);
    static vector<float> computeHistogram(caffe::Net<float> &CNN, const vector<vector<Sample> > &template_set, const vector<vector<Sample> > &test_set, vector<int> rotInv, vector<float> bins, int knn);
    static vector<vector<float>> computeConfusionMatrix(caffe::Net<float> &CNN,
                                                 const vector<vector<Sample>> &template_set,
                                                 const vector<vector<Sample>> &test_set, vector<string> models, unordered_map<string, int> local_index, int knn);
    static void saveLog(caffe::Net<float> &CNN, datasetManager &db, string config, int iter);
    static void saveConfusionMatrix(caffe::Net<float> &CNN, datasetManager &db, string config);
    static void computeManifold(caffe::Net<float> &CNN, const vector<vector<Sample> > &templates, int iter);
    static void visualizeKNN(caffe::Net<float> &CNN,
                             const vector<vector<Sample>> &test_set,
                             const vector<vector<Sample>> &templates);
};

}
