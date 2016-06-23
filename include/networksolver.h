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
#include "networkevaluator.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;
typedef networkEvaluator eval;

class networkSolver
{
public:
    networkSolver(string config, datasetManager *db);
    vector<Sample> buildBatch(int batch_size, int iter, bool bootstrapping);
    void trainNet(int resume_iter=0);
    void binarizeNet(int resume_iter=0);

    void computeKNN(caffe::Net<float> &CNN);
    void visualizeKNN(caffe::Net<float> &CNN, vector<string> test_models);
    bool bootstrap(caffe::Net<float> &CNN, int iter);

    void readParam(string config);
    void computeMaxSimTmpl();
    int getTmplVertex(int object, int pose){return templates[object][pose].label.at<float>(0,5);}
    int getTmplRot(int object, int pose){return templates[object][pose].label.at<float>(0,6);}

    hdf5Handler h5;
    std::random_device ran;
    vector<vector<vector<int>>> maxSimTmpl, maxSimKNNTmpl;
    unordered_map<string,int> model_index, global_model_index;
    vector<int> rotInv;
    unsigned int nr_objects, nr_training_poses, nr_template_poses, nr_test_poses;

    // Const references to db objects
    datasetManager *db;
    const vector<vector<int>>& vertex_tmpl;
    const vector<vector<Sample>>& templates, training_set, test_set;
    const vector<vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>>& tmpl_quats, training_quats, test_quats;

    // Config parameters
    vector<string> used_models, models;
    unsigned int num_epochs, num_training_rounds, binarization_epochs;
    unsigned int step_size;
    string network_path, net_name, learning_policy, binarization_net_name;
    float learning_rate, momentum, weight_decay, gamma;
    bool gpu, binarization, inplane;
};

#endif // NETWORKSOLVER_H
