#include <iostream>
#include <fstream>
#include "datasetmanager.h"
#include "networksolver.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

using namespace std;

int main(int argc, char *argv[])
{

    // Get config path
    if (argc < 2)
    {
        cerr << "Specify config file as argument" << endl;
        return 0;
    }
    string config(argv[1]);

<<<<<<< Updated upstream
    datasetManager db(config);
    db.createSceneSamplesAndTemplates();
    db.generateDatasets();
=======
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

    cout << network_path << endl;

    // Initialize the dataset_manager
    datasetManager db_manager(linemod_path, hdf5_path);
    // Generate the data set
    db_manager.createSceneSamplesAndTemplates(vector<string>({"ape","iron","cam"}));
>>>>>>> Stashed changes

    // Initialize the solver
    networkSolver solver(config, &db);

    // Train the network
    solver.trainNet(0);

    return 0;
}
