#include <iostream>
#include "datasetgenerator.h"
#include "networksolver.h"

using namespace std;

int main(int argc, char *argv[])
{
    // Define paths to the data
    string linemod_path = "/media/zsn/Storage/BMC/Master/Implementation/dataset/";
    string hdf5_path = "/media/zsn/Storage/BMC/Master/Implementation/WadimRestructured/hdf5/";
    string network_path = "/media/zsn/Storage/BMC/Master/Implementation/WadimRestructured/network/";

    // Initialize
    datasetGenerator generator(linemod_path, hdf5_path);
    networkSolver solver(network_path, hdf5_path);

    // Generate the data set
    generator.createSceneSamplesAndTemplates(vector<string>({"ape","bowl","cam"}));

    // Train the network online
//    solver.trainNet(vector<string>({"ape","bowl","cam"}));

    // Test the network
    // solv.testNet();
    // solv.testKNN(1);

    return 0;
}
