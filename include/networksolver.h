#ifndef NETWORKSOLVER_H
#define NETWORKSOLVER_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include "model.h"
#include "datatypes.h"
#include "hdf5handler.h"
#include "datasetmanager.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;

class networkSolver
{
public:
    networkSolver(string config, datasetManager *db);
    vector<Sample> buildBatch(int batch_size, int iter, bool bootstrapping);
    void trainNet(int resume_iter=0);
    void visualizeManifold(caffe::Net<float> &CNN, int iter);
    void visualizeKNN(caffe::Net<float> &CNN, vector<string> test_models);
    void computeKNN(caffe::Net<float> &CNN);
    void evaluateNetwork(caffe::Net<float> &CNN);
    bool bootstrap(caffe::Net<float> &CNN, int iter);
    Mat computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples);
private:
    hdf5Handler h5;
    std::random_device ran;
    datasetManager *db;
    vector<vector<vector<int>>> maxSimTmpl, maxSimKNNTmpl;
    unordered_map<string,int> model_index;
    unsigned int nr_objects, nr_training_poses, nr_template_poses, nr_test_poses;

    const vector<vector<Sample>>& training_set, test_set, templates;
    const vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>& tmpl_quats;
    const vector<vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>>& training_quats, test_quats;

    // Config parameters
    vector<string> used_models;
    unsigned int num_epochs, num_training_rounds;
    unsigned int step_size;
    string network_path, net_name, learning_policy;
    float learning_rate, momentum, weight_decay, gamma;
};

#endif // NETWORKSOLVER_H
