

#include <fstream>
#include <sstream>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "caffe/caffe.hpp"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "include/model.h"
#include "include/hdf5handler.h"
#include "include/datasetmanager.h"
#include "include/networksolver.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace boost;


#define SQR(x) ((x)*(x))

struct config
{
    // Define paths to the data
    string linemod_path,hdf5_path,network_path;
    bool GPU = false;
} CONFIG;



Matrix3f cam;
hdf5Handler h5;



float linemod_error(Isometry3f &gt, Isometry3f &pose, Model &m, string name)
{
    float error=0;
    if(name=="cup" || name=="bowl" || name=="eggbox" || name=="glue")
        for (Vector3f &p : m.getPoints())
        {
            float min_error = numeric_limits<float>::max();
            for (Vector3f &l : m.getPoints())
                min_error = min(min_error,(pose*p-gt*l).norm());
            error += min_error;
        }
    else for (Vector3f &p : m.getPoints()) error += (pose*p-gt*p).norm();
    return error / m.getPoints().size();
}




map<string,float> diameters,hashmod_hues;



void loadParams()
{
    assert(filesystem::exists(CONFIG.linemod_path+"diameters.txt"));
    ifstream dia(CONFIG.linemod_path+"diameters.txt");
    for (int i=0; i < 15; i++)
    {
        pair<string,float> temp;
        dia >> temp.first >> temp.second;
        diameters[temp.first] = temp.second;
    }

    diameters["ape"] *= 1.3f;
}


Mat showRGBDPatch(Mat &patch, bool show=true)
{
    vector<Mat> channels;
    cv::split(patch,channels);
    Mat RGB,D,out(patch.rows,patch.cols*2,CV_32FC3);
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));
    cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);
    D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));
    if (show){imshow("R G B D", out); waitKey();}
    return out;
}

// Takes a color and depth frame and samples a normalized 4-channel patch at the given center position and z-scale
Mat samplePatchWithScale(Mat &color, Mat &depth, int center_x, int center_y, float z, float fx, float fy)
{
    // Make a cut of metric size m
    float m = 0.2f;
    int screenW = fx * m/z;
    int screenH = fy * m/z;

    // Compute cut-out rectangle and ensure that we stay inside the image!
    Rect cut(center_x - screenW / 2, center_y - screenH / 2, screenW, screenH);
    if (cut.x < 0) cut.x = 0;
    if (cut.y < 0) cut.y = 0;
    if (cut.x > color.cols-screenW-1) cut.x = color.cols-screenW-1;
    if (cut.y > color.rows-screenH-1) cut.y = color.rows-screenH-1;

    assert (cut.x >= 0 && cut.x < color.cols-screenW);
    assert (cut.y >= 0 && cut.y < color.rows-screenH);

    // Cut-out from whole image
    Mat temp_col, temp_dep, final;
    color(cut).copyTo(temp_col);
    depth(cut).copyTo(temp_dep);

    Mat bg = (temp_dep == 0);

    // Convert to float and rescale to [-1,1]
    temp_col.convertTo(temp_col,CV_32FC3,1/255.f);
    temp_col = (temp_col-0.5f)*2.f;

    // Demean with central z value, clamp and rescale to [-1,1]
    temp_dep -= z;
    temp_dep.setTo(-m, temp_dep < -m);
    temp_dep.setTo(m, temp_dep > m);
    temp_dep *= 1.0f / m;

    // Resize
    const int CNN_INPUT_SIZE = 64;
    const Size final_size(CNN_INPUT_SIZE,CNN_INPUT_SIZE);
    resize(temp_col,temp_col,final_size);   // Standard bilinear interpolation
    resize(temp_dep,temp_dep,final_size,0,0,INTER_NEAREST);// Nearest-neighbor interpolation for depth!!!

    cv::merge(vector<Mat>{temp_col,temp_dep},final);

    // Bring all back to [0,1]
    final = (final+1.f)*0.5f;

    return final;
}




struct Hypo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Isometry3f pose;
    Mat col, dep, nor;
    pair<int,int> offset;
    int r, c;
    float screenW;
    //IcpStruct icp;
    float color_sim, depth_sim, nor_sim, linemod_error;
    int NNIdx;
    float distance;
    Hypo() : depth_sim(0), linemod_error(1000){}
};



struct SceneSample
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point pos2D;
    Vector3f pos3D;
    Mat patch, feat;
};

vector<SceneSample, Eigen::aligned_allocator<SceneSample> > sampleScene(Mat &col, Mat &dep, int step_size)
{
    int y_samples = (dep.rows / step_size)-1;
    int x_samples = (dep.cols / step_size)-1;

    // Collect samples
    vector<SceneSample, Eigen::aligned_allocator<SceneSample> > out;
    const int border = 3;
    for (int y=border; y <= y_samples-border; y++)
        for (int x=border; x <= x_samples-border; x++)
        {
            int gy = y*step_size, gx = x*step_size;
            if (y%2==0) gx += step_size/2;  // Add x-offset every odd row
            float d = dep.at<float>(gy,gx);
            if (d==0) continue;
            SceneSample s;
            s.patch = samplePatchWithScale(col,dep,gx,gy,d,cam(0,0),cam(1,1));
            s.pos2D.x = gx;
            s.pos2D.y = gy;
            out.push_back(s);
        }
    return out;
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
            binDescs.at<float>(r,b) = descs.at<float>(r,b) >= 0 ? 1 : 0;

        }
    return binDescs;
}

vector<Sample> extractSceneSamples(vector<Frame,Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index)
{
    vector<Sample> samples;
    for (Frame &f : frames)
    {

        // Instead of taking object centroid, take the surface point as central sample point
        Vector3f projCentroid = cam*f.gt.translation();
        projCentroid /= projCentroid(2);

        float z = f.depth.at<float>(projCentroid(1),projCentroid(0));
        assert(z>0.0f);

        Sample s;
        s.data = samplePatchWithScale(f.color,f.depth,projCentroid(0),projCentroid(1),z,cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion
        s.label = Mat(1,5,CV_32F);
        s.label.at<float>(0,0) = index;
        Quaternionf q(f.gt.linear());
        for (int i=0; i < 4; ++i)
            s.label.at<float>(0,1+i) = q.coeffs()(i);

        samples.push_back(s);
    }
    return samples;
}


vector<Sample> createTemplates(Model &model,Matrix3f &cam, int index, int subdiv)
{

    // Create synthetic views
    SphereRenderer sphere(cam);
    Vector3f scales(0.4, 1.1, 1.0);     // Render from 0.4 meters
    Vector3f in_plane_rots(-45,15,45);  // Render in_plane_rotations from -45 degree to 45 degree in 15degree steps
    vector<RenderView, Eigen::aligned_allocator<RenderView> > views =
            sphere.createViews(model,subdiv,scales,in_plane_rots,true,false,false);    // Equidistant sphere sampling with recursive level subdiv

    vector<Sample> samples;
    for (RenderView &v : views)
    {
        // Instead of taking object centroid, take the surface point as central sample point
        float z = v.dep.at<float>(cam(1,2),cam(0,2));
        assert(z>0.0f);

        Sample sample;
        sample.data = samplePatchWithScale(v.col,v.dep,cam(0,2),cam(1,2),z,cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion
        sample.label = Mat(1,5,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(v.pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);

        samples.push_back(sample);
    }
    return samples;

}



vector<Mat> computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples)
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
    vector<Mat> descs(samples.size());

    currIdxs.reserve(batchSize);
    for (size_t i=0; i < samples.size(); ++i)
    {
        // Collect indices of samples to be processed for this batch
        if (samples[i].data.empty()) continue;
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
            {
                descs[currIdxs[j]] = Mat(1,desc_dim,CV_32F);
                memcpy(descs[currIdxs[j]].data, out[0]->cpu_data() + j*desc_dim, desc_dim*sizeof(float));
            }

            currIdxs.clear(); // Throw away current batch
        }
    }
    return descs;
}



void read_config_file(char *file)
{

    namespace po = boost::program_options;

    // Define variables
    po::options_description desc("Options");
    desc.add_options()("linemod_path", po::value<std::string>(&CONFIG.linemod_path), "Path to LineMOD dataset");
    desc.add_options()("hdf5_path", po::value<std::string>(&CONFIG.hdf5_path), "Path to training data as HDF5");
    desc.add_options()("network_path", po::value<std::string>(&CONFIG.network_path), "Path to networks");
    desc.add_options()("gpu", po::value<bool>(&CONFIG.GPU), "GPU mode");

    // Read config file
    po::variables_map vm;
    std::ifstream settings_file(file);
    po::store(po::parse_config_file(settings_file , desc), vm);
    po::notify(vm);
}

Benchmark loadLinemodBenchmark(string linemod_path, string sequence, int count =-1)
{
    string dir_string = linemod_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color")
                last = std::max(last,std::stoi(file.substr(5,file.length())));

        }

    if (count>-1) last = count;
    Benchmark bench;
    for (int i=0; i <= last;i++)
    {
        Frame frame;
        frame.nr = i;
        frame.color = imread(dir_string + "/color"+to_string(i)+".jpg");
        frame.depth = imread(dir_string + "/inp/depth"+to_string(i)+".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);   // Bring depth map into meters
        ifstream pose(dir_string + "/pose"+to_string(i)+".txt");
        assert(pose.is_open());
        for (int k=0; k < 4;k++)
            for(int l=0; l < 4;l++)
                pose >> frame.gt.matrix()(k,l);
        bench.frames.push_back(frame);
    }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;
    return bench;
}


vector<string> models = {"ape","benchvise","cam","can","cat","driller","duck","holepuncher","iron","lamp","phone"};

void createSceneSamplesAndTemplates()
{

    for (int modelId = 0; modelId < models.size(); ++modelId)
    {

        string model_name = models[modelId];

        clog << "\nCreating samples and patches for " << model_name << ":" << endl;

        // - load model
        Model model;
        model.loadPLY(CONFIG.linemod_path + model_name + ".ply");

        // - load frames of benchmark and visualize
        Benchmark bench = loadLinemodBenchmark(CONFIG.linemod_path, model_name);

        // === Real data ===
        // - for each scene frame, extract RGBD sample
        vector<Sample> realSamples = extractSceneSamples(bench.frames,bench.cam,modelId);

        // - shuffle the samples
        random_shuffle(realSamples.begin(), realSamples.end());

        // - store realSamples to HDF5 files
        h5.write(CONFIG.hdf5_path + "realSamples_" + model_name +".h5", realSamples);
        //for (Sample &s : realSamples) showRGBDPatch(s.data);

        // === Synthetic data ===
        clog << "  - render synthetic data:" << endl;
        // - create synthetic samples and templates
        int subdivTmpl = 2; // sphere subdivision factor for templates
        vector<Sample> templates = createTemplates(model,bench.cam,modelId, subdivTmpl);
        vector<Sample> synthSamples = createTemplates(model,bench.cam,modelId, subdivTmpl+1);

        // - shuffle the samples
        //random_shuffle(templates.begin(), templates.end());
        random_shuffle(synthSamples.begin(), synthSamples.end());

        // - store realSamples to HDF5 files
        h5.write(CONFIG.hdf5_path + "templates_" + model_name + ".h5", templates);
        h5.write(CONFIG.hdf5_path + "synthSamples_" + model_name + ".h5", synthSamples);
        //for (Sample &s : templates) showRGBDPatch(s.data);

    }
}

/*
// Collect pixels equidistantly with valid depth data
samplePixels = sampleScene(color,chans[2],SAMPLE_STEP);

time_sampling = watch.restart();

// Sample local RGB-D patches
color.convertTo(float_col,CV_32FC3, 1/255.f);
samplePoints.clear();
samplePatches.clear();
for (Point &point : samplePixels)
{
    Mat patch = samplePatchWithScale(float_col,chans[2],point);
    if (patch.empty()) continue;
    samplePatches.push_back(patch);
    samplePoints.push_back(cloud.at<Vector3f>(point));
}

// Compute their CNN features
sampleFeats = runNet(samplePatches);

//recons = runNetReconstructions(samplePatches);
//for(size_t i=0; i<recons.size(); ++i)
//{
//    string dim = to_string(sampleFeats[0].cols);
//    imwrite("patch"+to_string(i)+".png",255*showRGBDPatch(samplePatches[i],false));
//    imwrite(dim+"/recon"+to_string(i)+".png",255*showRGBDPatch(recons[i],false));
//}

time_feats = watch.restart();


// Discard old probability maps
for (auto &m : probMaps) m.second.release();
vector<Vote6D, Eigen::aligned_allocator<Vote6D>> final_votes;
*/

int main(int argc, char *argv[])
{

    if (argc<2)
    {
        cerr << "Specifiy config file as argument" << endl;
        return 0;
    }
    read_config_file(argv[1]);

    //createSceneSamplesAndTemplates();


    if (CONFIG.GPU)  caffe::Caffe::set_mode(caffe::Caffe::GPU);

    string net_name = "manifold_wang_16";
    datasetManager manager(CONFIG.linemod_path,CONFIG.hdf5_path);
    networkSolver solver(models,CONFIG.network_path,CONFIG.hdf5_path,manager);


    //solver.trainNet(net_name,0);



    caffe::Net<float> net(CONFIG.network_path + net_name + ".prototxt",caffe::TEST);
    //net.CopyTrainedLayersFrom(CONFIG.network_path + net_name + "_iter_" + to_string(0) + ".caffemodel");

    string seq = "iron";

    Benchmark bench = loadLinemodBenchmark(CONFIG.linemod_path,seq,10);
    vector<Sample> tpls =  h5.read(CONFIG.hdf5_path + "templates_" + seq + ".h5");

    Mat DB = solver.computeDescriptors(net,tpls);
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    matcher->add(DB);

    cam = bench.cam;
    for (Frame &f :  bench.frames)
    {

        vector<SceneSample, Eigen::aligned_allocator<SceneSample> > scene_samples = sampleScene(f.color,f.depth,8);


        Mat temp;
        f.color.copyTo(temp);
        for (SceneSample &s : scene_samples)
        {
            if (s.patch.empty()) cv::circle(temp,s.pos2D,2,Scalar(0,0,255));
            else cv::circle(temp,s.pos2D,2,Scalar(0,255,0));
        }
        imshow("samples",temp); waitKey();

    }

    return 0;
}



/*
using Negative = pair<Rect,Isometry3f>;

int main(int argc, char *argv[])
{
    //loadParams();

    string sequence = "duck";
    float dist_threshold = 0.5f;
    int kNN = 1;
    bool bin = false;

    if (argc>1) sequence = string(argv[1]);
    if (argc>2) dist_threshold = std::stof(string(argv[2]));
    if (argc>3) bin = (string(argv[3]) == "bin");


    const float linemod_threshold = 0.1f*diameters[sequence];
    Benchmark bench = loadLinemodBenchmark(sequence);
    for (auto &s : bench.models) db.loadObject(s.first,s.second);
    cam = bench.cam;


    DetectionObject *obj = db.getObject(sequence);
    SphereRenderer renderer(cam);


    createTemplates(sequence,bin);
    BFMatcher matcher(cv::NORM_L2);
    if (bin)
    {
        cerr << "Binary!" << endl;
        matcher = BFMatcher(cv::NORM_HAMMING);
    }
    matcher.add(templates);


    string filename = DATA + sequence;
    if (bin) filename += "_bin";
    filename += ".h5";
    assert(filesystem::exists(filename));
    vector<int> scales;
    for (auto it : dims)
    {
        scales.push_back(it.first);
        scale_feats[it.first] = readDescriptorsHDF5(filename,it.first,bin);
        assert(templates.cols == scale_feats[it.first][0].cols);
        for (int rot=-15; rot <= 15; rot += 15)
            createViews(sequence,scales.back(),rot);
    }

    verification.init(&db,cam);


    int detected = 0, recalled=0;
    for (Frame &f : bench.frames)
    {
        depth2cloud(f.depth, f.cloud,cam);
        depth2normals(f.depth,f.normals,cam);
    }

    ofstream negfile("neg_"+sequence+".txt");
    for (Frame &f : bench.frames)
    {
        //  cerr << "Frame " << f.nr  << " ";




        vector<int> test_scales = scales;
#if 1  // Only fetch closest scale
        int nn_scale=0;
        int gt_z = std::round(f.gt[0].second.translation()(2)*100);
        for (int t : scales) if (std::abs(t-gt_z) < std::abs(nn_scale-gt_z)) nn_scale = t;
        test_scales = {nn_scale};
#endif

        vector<Hypo> frame_hypos;
        for (int scale : test_scales)
        {

            float scale_factor = getScaleFactorForDepth(scale*0.01f);
            //cerr << "GT_z: " << gt_z << " Scale: " << scale  << "  | Scale factor: " << scale_factor << endl;

            Mat &feats = scale_feats[scale][f.nr];

            float screenW = targetSize/scale_factor;
            int x_feats = dims[scale].first, y_feats = dims[scale].second;

            vector<vector<DMatch> > matches;
            matcher.knnMatch(feats,matches,kNN);

            for (int r=1; r < y_feats-1;++r)
                for (int c=1; c < x_feats-1;++c)
                {

                    int global_r = r*(max_pool_factor/scale_factor);
                    int global_c = c*(max_pool_factor/scale_factor);

                    for (DMatch &match : matches[c*y_feats + r])
                    {
                        if (match.distance> dist_threshold) continue;

                        DetectionView NN;

                        Hypo temp;

                        // ICP
                        Vector3f centroid(0,0,0), best_centroid(0,0,0);
                        vector<Vector3f> src, best_src;
                        for (int rot=-15; rot <= 15; rot += 15)
                        //for (int rot=0; rot <= 0; rot += 15)
                        {
                            NN = scale2views[rot][scale][match.trainIdx];
                            centroid.setZero();
                            src.clear();
                            for(size_t i=0; i < NN.ero_positions.size(); ++i)
                            {
                                Point &pos = NN.ero_positions[i];
                                Vector3f &t = f.cloud.at<Vector3f>(global_r+pos.y,global_c+pos.x);
                                if(std::abs(t(2)-NN.depths[i]) > 0.05f) continue;
                                centroid += t;
                                src.push_back(t);
                            }
                            if (src.size()<=best_src.size()) continue;
                            best_src = src;
                            best_centroid = centroid;
                        }
                        if (best_src.empty()) continue;

                        temp.pose = NN.pose;
                        temp.pose.translation() = (best_centroid/ (float) best_src.size()) + NN.offsetToCenter;
                        Isometry3f hypo_inv = temp.pose.inverse();
                        for (Vector3f &p : best_src) p = hypo_inv*p;
                        temp.icp = obj->icp.computeWithOutliers(best_src,5,0.0001f);
                        if (temp.icp.error > 0.01)  continue;

                        temp.pose = temp.pose*temp.icp.pose.inverse();
                        temp.r = global_r;
                        temp.c = global_c;
                        temp.screenW = screenW;
                        temp.NNIdx = match.trainIdx;
                        temp.distance = match.distance;
                        temp.offset = renderer.renderView(*obj,temp.pose,temp.col,temp.dep);

                        frame_hypos.push_back(temp);

#if 0
                        Mat out(480,2*640,CV_8UC3),col,lol;
                        f.color.copyTo(out(Rect(0,0,640,480)));
                        f.color.copyTo(out(Rect(640,0,640,480)));
                        rectangle(out,Rect(global_c,global_r,screenW,screenW),Scalar(0,0,255));
                        for(Point &pos : NN.ero_positions) out.at<Vec3b>(global_r+pos.y,global_c+pos.x) = Vec3b(0,255,0);
                        renderer.renderView(*obj,NN.pose,col,lol,false);
                        col(Rect(320-screenW/2,240-screenW/2,screenW,screenW)).copyTo(lol);
                        lol.copyTo(out(Rect(0,0,lol.cols,lol.rows)));
                        for(int r = 0; r < temp.dep.rows; ++r)
                            for(int c = 0; c < temp.dep.cols; ++c)
                                if (temp.dep.at<float>(r,c)>0)
                                    out.at<Vec3b>(r + temp.offset.second,c+temp.offset.first + 640) = Vec3b(0,255,0);
                        imshow("Frame NN after ICP",out);waitKey();
#endif


                    }
                }
        }


        // Run projective ICP rounds
        for (int run = 0; run < 3; run++)
        {
            tbb::parallel_for<uint>(0, frame_hypos.size(), [&] (uint i)
            {
                Hypo &h = frame_hypos[i];
                vector<Vector3f> src;
                Isometry3f hypo_inv = h.pose.inverse();
                for(int r = 0; r < h.dep.rows; ++r)
                    for(int c = 0; c < h.dep.cols; ++c)
                    {
                        float syn_d = h.dep.at<float>(r,c);
                        if (syn_d==0) continue;
                        int global_r = r+h.offset.second;
                        int global_c = c+h.offset.first;
                        if (global_r < 0 || global_r >= 480) continue;
                        if (global_c < 0 || global_c >= 640) continue;
                        Vector3f &t = f.cloud.at<Vector3f>(global_r,global_c);
                        if (std::abs(syn_d-t(2)) < 0.02f) src.push_back(hypo_inv*t);

                    }
                h.icp = obj->icp.computeWithOutliers(src,3,0.0001f);
                h.pose = h.pose*h.icp.pose.inverse();
            });

            for (Hypo &h : frame_hypos) h.offset = renderer.renderView(*obj,h.pose,h.col,h.dep);
        }



        tbb::parallel_for<uint>(0, frame_hypos.size(), [&] (uint i)
        {
            Hypo &h = frame_hypos[i];
            depth2normals(h.dep,h.nor,cam(0,0),cam(1,1),cam(0,2), cam(1,2));
            int sum = 0, counter=0;
            h.nor_sim=0;
            for(int r = 0; r < h.dep.rows; ++r)
                for(int c = 0; c < h.dep.cols; ++c)
                {
                    float syn_d = h.dep.at<float>(r,c);
                    if (syn_d==0) continue;
                    counter++;

                    int global_r = r+h.offset.second;
                    int global_c = c+h.offset.first;
                    if (global_r < 0 || global_r >= 480) continue;
                    if (global_c < 0 || global_c >= 640) continue;
                    if (std::abs(syn_d-f.depth.at<float>(global_r,global_c)) < 0.02f) sum++;
                    h.nor_sim += h.nor.at<Vector3f>(r,c).dot(f.normals.at<Vector3f>(global_r,global_c));
                }
            h.depth_sim = sum /(float) counter;
            h.nor_sim = h.nor_sim / (float) counter;
        });



        tbb::parallel_for<uint>(0,frame_hypos.size(),[&](uint i){frame_hypos[i].linemod_error=linemod_error(f.gt[0].second,frame_hypos[i].pose,*obj);});

        Hypo best;
        bool correct_among = false;
        vector<Negative> negatives;
        for (Hypo &h : frame_hypos)
        {
            if (h.depth_sim < 0.6f) continue;
            if (h.nor_sim < 0.6f) continue;

            correct_among |=  h.linemod_error <= linemod_threshold;

            if (h.linemod_error > linemod_threshold) negatives.push_back({Rect(h.c,h.r,h.screenW,h.screenW),h.pose});

            if (h.icp.error < best.icp.error)
                if (h.icp.inlier > best.icp.inlier)
                    best = h;
            //if (h.nor_sim > best.nor_sim) best = h;
        }

        if (correct_among) recalled++;
        //else cerr << "-";

        if (best.linemod_error <= linemod_threshold)
        //if (linemod_error(f.gt[0].second,best.pose,*obj) <= linemod_threshold)
        {
            //cerr << "*";
            detected++;
        }
#if 0
        else
        {
            cerr << "Inlier " << best.icp.inlier << endl;
            cerr << "Error " << best.icp.error << endl;
            cerr << "Depth " << best.depth_sim << endl;
            cerr << "Normals" << best.nor_sim << endl;
            Mat augm;
            verification.clearDetections();
            verification.addDetection(sequence,best.pose,false);
            verification.augment(f.color,augm);
            imshow("Frame",augm); waitKey();
        }
#endif


#if 1
        vector<Negative> filtered;
        for (Negative &n : negatives) if ((n.second.translation()-f.gt[0].second.translation()).norm() > 0.1) filtered.push_back(n);
        negfile << f.nr << endl << filtered.size() << endl;
        for (Negative &n : filtered) negfile << n.first.x << " " << n.first.y << " " << n.first.width << " " << n.first.height << endl;
#endif



        // cerr << endl;
    }

    cerr << sequence << " " << dist_threshold << " "  << kNN << endl;
    cerr << "Detected: " << detected << "  Ratio: " << 100.f * detected / (float) bench.frames.size() << endl;
    cerr << "Recalled: " << recalled << "  Ratio: " << 100.f * recalled / (float) bench.frames.size() << endl;



    return 0;

}

*/
