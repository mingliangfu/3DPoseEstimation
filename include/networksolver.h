#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include "model.h"
#include "datatypes.h"
#include "hdf5handler.h"
#include "datasetmanager.h"
#include "networkevaluator.h"
#include "helper.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;
typedef sz::networkEvaluator eval;


namespace sz { // For Wadim

class networkSolver
{
public:
    networkSolver(string config, datasetManager &db);
    void readParam(string config);

    void buildBatchQueue(size_t batch_size, size_t triplet_size, size_t epoch_iter,
                         size_t slice, size_t channels, size_t target_size, std::queue<vector<float>> &batch_queue);
    vector<Sample> buildBatch(int batch_size, unsigned int triplet_size, int iter);
    vector<Sample> buildBatchClass(int batch_size, unsigned int triplet_size, int iter);
    void trainNet(int resume_iter=0, bool threaded=true);
    void binarizeNet(int resume_iter=0);

    void computeKNN(caffe::Net<float> &CNN);
    bool bootstrap(caffe::Net<float> &CNN, int iter);

    hdf5Handler h5;
    std::random_device ran;
    vector<vector<vector<int>>> maxSimKNNTmpl;
    unordered_map<string,int> model_index, global_model_index;
    vector<int> rotInv;
    unsigned int nr_objects, nr_training_poses, nr_template_poses, nr_test_poses;

    // Thread variables
    size_t thread_iter;
    std::condition_variable cond;
    std::mutex queue_mutex;

    // Const references to db objects
    datasetManager &db;
    const vector<vector<Sample>>& template_set, training_set, test_set;
    const vector<vector<vector<int>>>& maxSimTmpl;

    // Config parameters
    bool bootstrapping;
    vector<string> used_models, models;
    unsigned int num_epochs, binarization_epochs, num_training_rounds, num_bootstrapping_rounds, log_epoch;
    unsigned int step_size;
    string config, network_path, net_name, learning_policy, binarization_net_name;
    float learning_rate, momentum, weight_decay, gamma;
    bool gpu, binarization, inplane;
    int random_background;
};

}
