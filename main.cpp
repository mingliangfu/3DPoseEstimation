#include <iostream>
#include <fstream>
#include "datasetmanager.h"
#include "networksolver.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

using namespace std;
using namespace sz;

int main(int argc, char *argv[])
{

    // Get config path
    if (argc < 2)
    {
        cerr << "Specify config file as argument" << endl;
        return 0;
    }
    string config(argv[1]);

    datasetManager db(config);
    db.createSceneSamplesAndTemplates();
    db.generateDatasets();

    // Initialize the solver
    networkSolver solver(config, &db);

    // Train the network
    solver.trainNet(0);

    return 0;
}
