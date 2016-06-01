#include <iostream>
#include <fstream>
#include "datasetgenerator.h"
#include "networksolver.h"

#include "boost/program_options.hpp"


using namespace std;

namespace po = boost::program_options;


// Define paths to the data
string linemod_path = "/media/zsn/Storage/BMC/Master/Implementation/dataset/";
string hdf5_path = "/media/zsn/Storage/BMC/Master/Implementation/WadimRestructured/hdf5/";
string network_path = "/home/zsn/Documents/3DPoseEstimation/network/";
bool GPU = false;



int main(int argc, char *argv[])
{

    if (argc<2)
    {
        cerr << "Specifiy config file as argument" << endl;
        return 0;
    }


    // Define variables
    po::options_description desc("Options");
    desc.add_options()("linemod_path", po::value<std::string>(&linemod_path), "Path to LineMOD dataset");
    desc.add_options()("hdf5_path", po::value<std::string>(&hdf5_path), "Path to training data as HDF5");
    desc.add_options()("network_path", po::value<std::string>(&network_path), "Path to networks");
    desc.add_options()("gpu", po::value<bool>(&GPU), "GPU mode");

    // Read config file
    po::variables_map vm;
    std::ifstream settings_file(argv[1]);
    po::store(po::parse_config_file(settings_file , desc), vm);
    po::notify(vm);

    // Initialize
    datasetGenerator generator(linemod_path, hdf5_path);
    networkSolver solver(network_path, hdf5_path);

    // Generate the data set
    //generator.createSceneSamplesAndTemplates(vector<string>({"ape","iron","cam"}));


    if (GPU)
        caffe::Caffe::set_mode(caffe::Caffe::GPU);


    // Train the network online
    //solver.trainNetWang(vector<string>({"ape","iron","cam"}), "manifold_wang", 0);

    // Test the network
     solver.testNet();
     solver.testKNN(vector<string>({"ape","iron","cam"}));

    return 0;
}
