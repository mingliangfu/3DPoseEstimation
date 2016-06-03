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
    datasetManager(string dataset_path, string hdf5_path);
    Benchmark loadLinemodBenchmark(string linemod_path, string sequence, int count=-1);
    Mat samplePatchWithScale(Mat &color, Mat &depth, int center_x, int center_y, float z, float fx, float fy);
    vector<Sample> extractSceneSamplesPaul(vector<Frame, Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index);
    vector<Sample> extractSceneSamplesWadim(vector<Frame, Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index);
    vector<Sample> createTemplatesPaul(Model &model, Matrix3f &cam, int index);
    vector<Sample> createTemplatesWadim(Model &model, Matrix3f &cam, int index, int subdiv);
    void createSceneSamplesAndTemplates(vector<string> used_models);
    void saveSamples();
    void generateDatasets(vector<string> used_models, vector<vector<Sample>>& trainingSet, vector<vector<Sample>>& testSet, vector<vector<Sample>>& templates);
    void addNoiseToSynthData(int copies, vector<vector<Sample>>& trainingSet);
private:
    string dataset_path, hdf5_path, network_path;
    vector<string> models;
    unordered_map<string,int> model_index;
    hdf5Handler h5;

};

#endif // DATASETMANAGER_H
