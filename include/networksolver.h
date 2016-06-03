#ifndef NETWORKSOLVER_H
#define NETWORKSOLVER_H

#include <iostream>
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
#include "datasetmanager.h"

class networkSolver
{
public:
    networkSolver(vector<string> used_models, string network_path, string hdf5_path, datasetManager db_manager);
    vector<Sample> buildBatch(int batch_size, int iter);
    void setNetworkParameters();
    void trainNet(string net_name, int resume_iter=0);
    void testManifold(string net_name, int resume_iter);
    void testKNN(string net_name, int resume_iter, vector<string> test_models);
    void shuffleTrainingSet();
    Mat computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples);
    Mat showRGBDPatch(Mat &patch, bool show=true);
private:
    hdf5Handler h5;
    string network_path;
    string hdf5_path;
    std::random_device ran;
    vector<string> used_models;
    vector<vector<Sample>> training_set, test_set, templates;
    // Build a bool vector for each object that stores if all templates have been used yet
    vector<vector<bool>> training_used;
    vector< vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > > training_quats;
    vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>> tmpl_quats;
    datasetManager db_manager;
    unordered_map<string,int> model_index;
};

#endif // NETWORKSOLVER_H
