#ifndef DATASETMANAGER_H
#define DATASETMANAGER_H

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

#include <fstream>
#include <sstream>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "sphere.h"
#include "datatypes.h"
#include "hdf5handler.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;

class datasetManager
{
public:
    datasetManager(string config);
    Benchmark loadLinemodBenchmark(string linemod_path, string sequence, int count=-1);
    Mat samplePatchWithScale(Mat &color, Mat &depth, int center_x, int center_y, float z, float fx, float fy);
    vector<Sample> extractSceneSamplesPaul(vector<Frame, Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index);
    vector<Sample> extractSceneSamplesWadim(vector<Frame, Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index);
    vector<Sample> createTemplatesPaul(Model &model, Matrix3f &cam, int index, int rotInv);
    vector<Sample> createTemplatesWadim(Model &model, Matrix3f &cam, int index, int rotInv, int subdiv);
    void createSceneSamplesAndTemplates();
    void saveSamples();
    void generateDatasets();
    void computeQuaternions();
    void addNoiseToSynthData(unsigned int copies, vector<vector<Sample>>& trainingSet);

    const vector<vector<Sample>>& getTrainingSet() const {return training_set;}
    const vector<vector<Sample>>& getTemplateSet() const {return templates;}
    const vector<vector<Sample>>& getTestSet() const {return training_set;}
    const vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>& getTmplQuats() const {return tmpl_quats;}
    const vector<vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>>& getTrainingQuats() const {return training_quats;}
    const vector<vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>>& getTestQuats() const {return test_quats;}

    int getTrainingSetSize() {return training_set[0].size();}
    int getTemplateSetSize() {return templates[0].size();}
    int getTestSetSize() {return test_set[0].size();}
    int getNrObjects() {return used_models.size();}


private:
    vector<vector<Sample>> training_set, test_set, templates;
    vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>> tmpl_quats;
    vector<vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>> training_quats, test_quats;
    unsigned int nr_objects, nr_training_poses, nr_template_poses, nr_test_poses;

    string dataset_path, hdf5_path;
    vector<string> models, used_models;
    unordered_map<string,int> model_index;
    vector<int> rotInv;
    hdf5Handler h5;

};

#endif // DATASETMANAGER_H
