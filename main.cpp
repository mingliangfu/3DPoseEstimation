#include <iostream>
#include <fstream>
#include <bitset>

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


Mat computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples)
{

    caffe::Blob<float>* input_layer = CNN.input_blobs()[0];
    const size_t batchSize = input_layer->num();
    const int channels =  input_layer->channels();
    const int targetSize = input_layer->height();
    const int slice = input_layer->height()*input_layer->width();
    const int img_size = slice*channels;

    caffe::Blob<float>* output_layer = CNN.output_blobs()[0];
    const size_t desc_dim = output_layer->channels();

    vector<float> data(batchSize*img_size,0);
    vector<int> currIdxs;
    Mat descs(samples.size(),desc_dim,CV_32F);

    currIdxs.reserve(batchSize);
    for (size_t i=0; i < samples.size(); ++i)
    {
        // Collect indices of samples to be processed for this batch
        currIdxs.push_back(i);
        if (currIdxs.size() == batchSize || i == samples.size()-1) // If batch full or last sample
        {
            // Fill linear batch memory with input data in Caffe layout with channel-first
            for (size_t j=0; j < currIdxs.size(); ++j)
            {
                Mat &patch = samples[currIdxs[j]].data;
                int currImg = j*img_size;
                for (int ch=0; ch < channels ; ++ch)
                    for (int y = 0; y < targetSize; ++y)
                        for (int x = 0; x < targetSize; ++x)
                            data[currImg + slice*ch + y*targetSize + x] = patch.ptr<float>(y)[x*channels + ch];
            }
            // Copy data memory into Caffe input layer, process batch and copy result back
            input_layer->set_cpu_data(data.data());
            vector< caffe::Blob<float>* > out = CNN.Forward();

            for (size_t j=0; j < currIdxs.size(); ++j)
                memcpy(descs.ptr<float>(currIdxs[j]), out[0]->cpu_data() + j*desc_dim, desc_dim*sizeof(float));

            currIdxs.clear(); // Throw away current batch
        }
    }
    return descs;
}


Mat binarizeDescriptors(Mat &descs)
{

    Mat binDescs = Mat::zeros(descs.size(),CV_32F);
    for (int r=0; r < descs.rows; ++r)
        for (int b=0; b < descs.cols; ++b)
        {
            //int curr_byte = b/8;
            //int curr_bit = b - (curr_byte*8);
            //binDescs.at<uchar>(r,curr_byte) |= (descs.at<float>(r,b) >= 0) << curr_bit;
            //cerr << descs.at<float>(r,b) << " " <<std::bitset<8>(binDescs.at<uchar>(r,curr_byte))<<std::endl;
            //binDescs.at<float>(r,b) = descs.at<float>(r,b) >= 0 ? 1 : 0;
            binDescs.at<float>(r,b) = descs.at<float>(r,b);

        }
    return binDescs;
}

Mat showRGBDPatch(Mat &patch)
{
    vector<Mat> channels;
    cv::split(patch,channels);
    Mat RGB,D,out(patch.rows,patch.cols*2,CV_32FC3);
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));
    cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);
    D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));
    return out;
}

void evaluateNetwork(string net_name, int iter, string seq, vector<string> DB, bool binary)
{

    hdf5Handler h5;


    // Load the snapshot
    caffe::Net<float> CNN(network_path + net_name + ".prototxt", caffe::TEST);
    CNN.CopyTrainedLayersFrom(net_name + "_iter_" + to_string(iter)+ ".caffemodel");

    // Get the test data
    Mat DBfeats, DBtest;
    vector<Sample> real = h5.read(hdf5_path + "realSamples_" + seq + ".h5"), tmpl;

    for (string s : DB)
    {
        vector<Sample> temp = h5.read(hdf5_path + "templates_" + s + ".h5");
        for (Sample &s : temp) tmpl.push_back(s);
    }


    DBtest.push_back(computeDescriptors(CNN, real));
    DBfeats.push_back(computeDescriptors(CNN, tmpl));

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher;

    if (binary)
    {
         //matcher = DescriptorMatcher::create("BruteForce-Hamming");
         matcher = DescriptorMatcher::create("BruteForce");
         DBtest = binarizeDescriptors(DBtest);
         DBfeats = binarizeDescriptors(DBfeats);
    }
    else matcher = DescriptorMatcher::create("BruteForce");

    matcher->add(DBfeats);


    vector< vector<DMatch> > matches;
    int knn=1;
    matcher->knnMatch(DBtest, matches, knn);

    vector<float> bins = {-1, 5, 15, 30, 45, 90,180};
    vector<float> histo(bins.size(),0);

    for (int i=0; i < DBtest.rows; ++i)
        for (DMatch &m : matches[i])
        {

            Mat query = showRGBDPatch(real[m.queryIdx].data);
            Mat nn = showRGBDPatch(tmpl[m.trainIdx].data);
            cerr << DBtest.row(m.queryIdx) << endl;
            cerr << DBfeats.row(m.trainIdx) << endl;

            imshow("query",query); imshow("nn",nn); waitKey();




            if (real[m.queryIdx].label.at<float>(0,0) != tmpl[m.trainIdx].label.at<float>(0,0))
            {
                histo[0]++;
                continue;
            }

            Quaternionf sampleQuat, kNNQuat;
            for (int q=0; q < 4; ++q)
            {
                sampleQuat.coeffs()(q) = real[m.queryIdx].label.at<float>(0,1+q);
                kNNQuat.coeffs()(q) = tmpl[m.trainIdx].label.at<float>(0,1+q);
            }

            float diff =  sampleQuat.angularDistance(kNNQuat)*180.f/M_PI;
            for (size_t b = 1; b < bins.size(); ++b)
                if (diff < bins[b])
                {
                    histo[b]++;
                    break;
                }

        }

    float total=0;
    for (float &i : histo) total += i;
    for (float &i : histo) i /= total;
    for (size_t i=0; i < histo.size(); ++i)
        cerr << i << " " << histo[i] << endl;


}





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
    //solver.trainNetWang(vector<string>({"ape","iron","cam"}), "manifold_wang_64_bin", 0);

    // Test the network
    //solver.testNet();
    //solver.testKNN(vector<string>({"ape","iron","cam"}));

    //evaluateNetwork("manifold_wang_64",25000,"cam",vector<string>({"ape","iron","cam"}),false);
    evaluateNetwork("manifold_wang_64_bin",25000,"cam",vector<string>({"ape","iron","cam"}),true);

    return 0;
}
