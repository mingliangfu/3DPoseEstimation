#pragma once

#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/photo/photo.hpp>

#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "sphere.h"
#include "datatypes.h"
#include "hdf5handler.h"
#include "helper.h"
#include "bgfill.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;

namespace sz {

class datasetManager
{
public:
    datasetManager(string config);
    Benchmark loadLinemodBenchmark(string linemod_path, string sequence, int count=-1);
    Benchmark loadBigbirdBenchmark(string bigbird_path, string sequence, int count=-1);
    Benchmark loadWashingtonBenchmark(string washington_path, string sequence, int count=-1);
    Benchmark loadBenjaminBenchmark(string benjamin_path, string sequence);
    Mat samplePatchWithScale(Mat &color, Mat &depth, Mat &normals, int center_x, int center_y, float z, float fx, float fy);
    vector<Sample> extractRealSamplesPaul(vector<Frame> &frames, Matrix3f &cam, int index, Model &model);
    vector<Sample> extractRealSamplesWadim(vector<Frame> &frames, Matrix3f &cam, int index);
    vector<Sample> createSynthSamplesPaul(Model &model, Matrix3f &cam, int index);
    vector<Sample> createSynthSamplesWadim(Model &model, Matrix3f &cam, int index, int subdiv);
    vector<Sample> extractSynthSamplesBenjamin(string benjamin_path, Matrix3f &cam, string sequence, int index);


    void loadLinemodHardNegatives();

    void generateAndStoreSamples(int sampling_type); // 0 - Paul, 1 - Wadim
    void generateDatasets();
    void computeMaxSimTmplInplane();
    void computeMaxSimTmpl();
    void randomFill(Mat &patch, int type);

    // Helper methods
    const vector<vector<Sample>>& getTrainingSet() const {return training_set;}
    const vector<vector<Sample>>& getTemplateSet() const {return template_set;}
    const vector<vector<Sample>>& getTestSet() const {return test_set;}
    const vector<vector<vector<int>>>& getMaxSimTmpl() const {return maxSimTmpl;}

    // Keep your fingers away from these, Sergey!
    const vector<string>& getModels() const {return used_models;}
    const unordered_map< string, vector<Sample> >& getHardNegatives() const {return hard_negatives;}
    string getDatasetPath() const {return dataset_path;}
    string getHDF5Path() const {return hdf5_path;}

private:
    std::random_device ran;
    vector<vector<Sample>> template_set, training_set, test_set;
    unsigned int nr_objects;
    vector<vector<vector<int>>> maxSimTmpl;

    string dataset_path, hdf5_path, bg_path, dataset_name;
    string simulated_templates_path, simulated_test_set_path;
    vector<string> models, used_models;
    unordered_map<string,int> model_index;
    vector<int> rotInv;
    bool inplane, use_real, use_simulated;
    hdf5Handler h5;

    unordered_map<string, vector<Sample> > hard_negatives;

    int random_background;
    bgfill bg;

};

}

