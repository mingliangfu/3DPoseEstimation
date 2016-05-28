#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <H5Cpp.h>

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"

#include <boost/filesystem.hpp>

#include "sphere.h"


using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;


vector<string> models = {"ape","benchvise","bowl","cam","can","cat", "cup","driller",
                         "duck","eggbox","glue","holepuncher","iron","lamp","phone"};

unordered_map<string,int> model_index;
string LINEMOD_path = "/media/zsn/Storage/BMC/Master/Implementation/dataset/";
string hdf5_path = "/media/zsn/Storage/BMC/Master/Implementation/Wadim/build/debug/hdf5/";
string network_path = "/media/zsn/Storage/BMC/Master/Implementation/Wadim/network/";

std::random_device ran;


static inline void loadbar( string label, unsigned int x, unsigned int n, unsigned int w = 20)
{
    if ( (x != n) && (x % (n/100+1) != 0) ) return;

    float ratio  =  x/(float)n;
    int   c      =  ratio * w;

    clog << label << setw(3) << (int)(ratio*100) << "% [";
    for (int x=0; x<c; x++) clog << "=";
    for (int x=c; x<w; x++) clog << " ";
    clog << "]\r" << flush;

    if ( x == n ) clog << endl;
}


struct Frame
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int nr;
    Mat color, depth, cloud, mask;
    Isometry3f gt;
};

struct Benchmark
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vector<Frame, Eigen::aligned_allocator<Frame> > frames;
    Matrix3f cam;
};


Benchmark loadLinemodBenchmark(string LINEMOD_path, string sequence, int count=-1)
{
    string dir_string = LINEMOD_path + sequence;
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
//    cout << "  - loading frames in the range " << 0 << " - " << last << ":"<< endl;
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
//        cout << sequence << ", sample:  " << i << endl;
        loadbar("  - loading frames: ",i,last);
    }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;
    return bench;
}

struct Sample
{
    Mat data, label;
};

// Define triplet struct and container
struct Triplet{Sample anchor, puller, pusher;};
// Define pairs
struct Pair{Sample anchor, puller;};

struct TripletsPairs
{
    vector<Triplet> triplets;
    vector<Pair> pairs;
};

void showGT(Benchmark &bench, Model &m)
{
    Matrix3f cam = bench.cam;
    Mat out;
    for (Frame &fr : bench.frames)
    {
        cerr << "Frame #" << fr.nr << endl;

        fr.color.copyTo(out);
        Painter painter;
        BackgroundCamera bcamera(out);
        RealWorldCamera c(cam,fr.gt);
        BoundingBox bb(m.boundingBox);

        painter.clearObjects();
        painter.setBackground(0,0,0);
        painter.addPaintObject(&bcamera);
        painter.addPaintObject(&c);
        painter.addPaintObject(&bb);
        painter.addPaintObject(&m);
        painter.paint();
        painter.copyColorTo(out);
        imshow("GT", out);
        waitKey();
    }
}



void writeHDF5(string filename, vector<Sample> &samples)
{
    if (samples.empty())
    {
        cerr << "writeHDF5: Nothing to write!" << endl;
        return;
    }

    try
    {
        // The HDF5 data layer in Caffe expects the following (everything float!):
        // Data must be Samples x Channels x Height x Width
        // Labels must be Samples x FeatureDim
        vector<hsize_t> data_dims = {samples.size(),(hsize_t) samples[0].data.channels(),(hsize_t) samples[0].data.rows,(hsize_t) samples[0].data.cols};

        // Specify size and shape of subset to write. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> slab_size = {1, data_dims[1], data_dims[2],data_dims[3]};

        H5::DataSpace data_space(data_dims.size(),data_dims.data());
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DataSet data_set = file.createDataSet("data",H5::PredType::NATIVE_FLOAT,data_space);

        // Fill temporary memory how Caffe needs it by filling the correct hyperslab per sample in the HDF5
        vector<float> temp(data_dims[1]*data_dims[2]*data_dims[3]);
        for(uint i=0; i < samples.size(); ++i)
        {
            for (hsize_t ch = 0; ch < data_dims[1]; ++ch)
                for (hsize_t y = 0; y < data_dims[2]; ++y)
                    for (hsize_t x = 0; x < data_dims[3]; ++x)
                        temp[ch*data_dims[2]*data_dims[3] + y*data_dims[3] + x] = samples[i].data.ptr<float>(y)[x*data_dims[1] + ch];

            offset[0] = i;
            data_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            data_set.write(temp.data(), H5::PredType::NATIVE_FLOAT, memspace, data_space);
        }

        vector<hsize_t> label_dims = {samples.size(), (hsize_t) samples[0].label.cols};
        H5::DataSpace label_space(label_dims.size(),label_dims.data());
        H5::DataSet label_set = file.createDataSet("label",H5::PredType::NATIVE_FLOAT,label_space);
        slab_size = {1, label_dims[1]};
        memspace = H5::DataSpace(slab_size.size(), slab_size.data());
        for(uint i=0; i < samples.size(); ++i)
        {
            offset[0] = i;
            label_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            label_set.write(samples[i].label.data, H5::PredType::NATIVE_FLOAT, memspace, label_space);
        }

    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}

vector<Sample> readHDF5(string filename)
{
    vector<Sample> samples;
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet data_set = file.openDataSet("data");
        H5::DataSet label_set = file.openDataSet("label");

        H5::DataSpace data_space = data_set.getSpace();
        H5::DataSpace label_space = label_set.getSpace();

        vector<hsize_t> data_dims(data_space.getSimpleExtentNdims());
        vector<hsize_t> label_dims(label_space.getSimpleExtentNdims());
        data_space.getSimpleExtentDims(data_dims.data(), nullptr);
        label_space.getSimpleExtentDims(label_dims.data(), nullptr);

        assert(data_dims[0] == label_dims[0]); // Make sure that data count = label count

        // Specify size and shape of subset to read. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> slab_size = {1, data_dims[1], data_dims[2],data_dims[3]};
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        // Copy from Caffe layout back into OpenCV layout
        vector<float> temp(data_dims[1]*data_dims[2]*data_dims[3]);   // Memory for one patch
        samples.resize(data_dims[0]);
        for (uint i=0; i < samples.size(); ++i)
        {
            // Select correct patch as hyperslab inside the HDF5 file
            offset[0] = i;
            data_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            data_set.read(temp.data(), H5::PredType::NATIVE_FLOAT, memspace, data_space);

            samples[i].data = Mat(data_dims[2],data_dims[3],CV_32FC(data_dims[1]));
            for (hsize_t chan=0; chan < data_dims[1]; ++chan)
            {
                auto currChan = chan*data_dims[2]*data_dims[3];
                for (hsize_t y =0; y < data_dims[2]; ++y)
                    for (hsize_t x =0; x < data_dims[3]; ++x)
                        samples[i].data.ptr<float>(y)[x*data_dims[1] + chan] = temp[currChan + y*data_dims[3] + x];
            }
        }

        slab_size = {1, label_dims[1]};
        memspace = H5::DataSpace(slab_size.size(), slab_size.data());
        for (uint i=0; i < samples.size(); ++i)
        {
            // Select correct label as hyperslab inside the HDF5 file
            samples[i].label = Mat(1,label_dims[1],CV_32F);
            offset[0] = i;
            label_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            label_set.read(samples[i].label.data, H5::PredType::NATIVE_FLOAT, memspace, label_space);
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
    return samples;
}


Mat showRGBDPatch(Mat &patch, bool show=true)
{
    vector<Mat> channels;
    //cv::split((patch+1.f)*0.5f,channels);
    cv::split(patch,channels);

    Mat RGB,D,out(patch.rows,patch.cols*2,CV_32FC3);

    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));

    cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);
    D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));

    if(show) {imshow("R G B D",out); waitKey();}
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



vector<Sample> extractSceneSamplesPaul(vector<Frame,Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index)
{
    vector<Sample> samples;
    for (Frame &f : frames)
    {
        Vector3f centroid = f.gt.translation();
        Vector3f projCentroid = cam*centroid;
        projCentroid /= projCentroid(2);

        Sample s;
        s.data = samplePatchWithScale(f.color,f.depth,projCentroid(0),projCentroid(1),centroid(2),cam(0,0),cam(1,1));

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



vector<Sample> createTemplatesPaul(Model &model,Matrix3f &cam, int index)
{

    ifstream file(LINEMOD_path+"paul/camPositionsElAz.txt");
    assert(file.good());
    vector<Vector2f, Eigen::aligned_allocator<Vector2f> > sphereCoords(1542);

    // Read all camera poses from file (given in elevation and azimuth)
    vector<Matrix3f, Eigen::aligned_allocator<Matrix3f> > camPos;
    for(Vector2f &v : sphereCoords)
    {
        file >> v(0) >> v(1);

        // Copied from basetypes.py lines 100-xxx
        // To move into el=0,az=0 position we first have to rotate 45deg around x
        AngleAxisf camRot0(M_PI/2,Vector3f(1,0,0));

        // Build rotation matrix from spherical coordinates
        AngleAxisf el(v(0),Vector3f(1,0,0));
        AngleAxisf az(-v(1),Vector3f(0,0,1));
        Matrix3f camRot = (el*az).toRotationMatrix();
        camPos.push_back(camRot0*camRot);
    }

    // Retain only the 301 template poses
    //camPos.resize(301);

    // Render each and create proper sample
    SphereRenderer renderer;
    renderer.init(cam);
    Mat col,dep;
    Isometry3f pose = Isometry3f::Identity();
    pose.translation()(2) = 0.4f;

    vector<Sample> samples;
    for (Matrix3f &m : camPos)
    {
        pose.linear() = m;
        renderer.renderView(model,pose,col,dep,false);

        Sample sample;
        sample.data = samplePatchWithScale(col,dep,cam(0,2),cam(1,2),pose.translation()(2),cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion
        sample.label = Mat(1,5,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);

        samples.push_back(sample);
    }
    return samples;

}

vector<Sample> extractSceneSamplesWadim(vector<Frame,Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index)
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

vector<Sample> createTemplatesWadim(Model &model,Matrix3f &cam, int index, int subdiv)
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
        //        assert(z>0.0f);

        Sample sample;
        sample.data = samplePatchWithScale(v.col,v.dep,cam(0,2),cam(1,2),v.pose.translation()(2),cam(0,0),cam(1,1));

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



void createSceneSamplesAndTemplates(string seq)
{

    clog << "\nCreating samples and patches for " << seq << ":" << endl;

    // - load model
    Model model;
    model.loadPLY(LINEMOD_path+seq+".ply");

    // - load frames of benchmark and visualize
    Benchmark bench = loadLinemodBenchmark(LINEMOD_path,seq);
    //showGT(bench,model);

    // === Real data ===

    // - for each scene frame, extract RGBD sample
    vector<Sample> realSamples = extractSceneSamplesWadim(bench.frames,bench.cam,model_index[seq]);
    // vector<Sample> realSamples = extractSceneSamplesPaul(bench.frames,bench.cam,model_index[seq]);

    // - store realSamples to HDF5 files
    writeHDF5(hdf5_path + "realSamples_" + seq +".h5", realSamples);
    //    for (Sample &s : realSamples) showRGBDPatch(s.data);

    // === Synthetic data ===

    clog << "  - render synthetic data:" << endl;
    // - create synthetic samples and templates
    int subdivTmpl = 2; // sphere subdivision factor for templates
    vector<Sample> templates = createTemplatesWadim(model,bench.cam,model_index[seq], subdivTmpl);
    vector<Sample> synthSamples = createTemplatesWadim(model,bench.cam,model_index[seq], subdivTmpl+1);
    // vector<Sample> templates = createTemplatesPaul(model,bench.cam,model_index[seq]);

    // store realSamples to HDF5 files
    writeHDF5(hdf5_path + "templates_" + seq + ".h5", templates);
    writeHDF5(hdf5_path + "synthSamples_" + seq + ".h5", synthSamples);
    //for (Sample &s : templates) showRGBDPatch(s.data);


}

void buildTriplets(vector<string> used_models)
{

    // Define triplet struct and container
    vector<Triplet> triplets;

    size_t nr_objects = used_models.size();
    assert(nr_objects > 1);

    // Read stored templates and scene samples for the used models
    vector< vector<Sample> > templates, scene_samples;
    for (string &seq : used_models)
    {
        scene_samples.push_back(readHDF5(hdf5_path + "scenesamples_" + seq + ".h5"));
        templates.push_back(readHDF5(hdf5_path + "tempwadim_" + seq + ".h5"));
    }

    // Read quaternion poses from scene samples
    vector< vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > > scene_quats(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
    {
        scene_quats[i].resize(scene_samples[i].size());
        for (size_t k=0; k < scene_quats[i].size(); ++k)
            for (int j=0; j < 4; ++j)
                scene_quats[i][k].coeffs()(j) = scene_samples[i][k].label.at<float>(0,1+j);
    }


    // Read quaternion poses from templates (they are identical for all objects)
    int nr_poses = 2359;
    vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > tpl_quat(nr_poses);
    for  (int i=0; i < nr_poses; ++i)
        for (int j=0; j < 4; ++j)
            tpl_quat[i].coeffs()(j) = templates[0][i].label.at<float>(0,1+j);



    // Build a bool vector for each object that stores if all templates have been used yet
    vector< vector<bool> > tmpl_used(nr_objects);
    for (auto &vec : tmpl_used)
        vec.assign(nr_poses,false);

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_poses-1);

    // Random generators that returns a random scene sample for a given object
    vector< std::uniform_int_distribution<size_t> > ran_scene_sample(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
        ran_scene_sample[i] = std::uniform_int_distribution<size_t>(0, scene_samples[i].size()-1);

    cout << "Start building triplets!" << endl;

    for(bool finished=false; !finished;)
    {
        size_t anchor, puller, pusher;

        // Build each triplet by cycling through each object
        for (size_t obj=0; obj < nr_objects; obj++)
        {

            /// Type 0: A random scene sample together with closest template against another template

            // Pull random scene sample and find closest pose neighbor from templates
            size_t ran_sample = ran_scene_sample[obj](ran);
            puller=0;
            float best_dist = numeric_limits<float>::max();
            for (size_t temp=0; temp < nr_poses; temp++)
            {
                if (tmpl_used[obj][temp]) continue;  // Skip if template already used
                float temp_dist = scene_quats[obj][ran_sample].angularDistance(tpl_quat[temp]);
                if (temp_dist >= best_dist) continue;
                puller = temp;
                best_dist = temp_dist;
            }

            // TODO: what if closest unused template is actually rather far away in pose space? Better discard..?!

            // Mark template as used
            tmpl_used[obj][puller] = true;


            // Randomize through until pusher != puller
            pusher = ran_tpl(ran);
            while (pusher == puller) pusher = ran_tpl(ran);

            Triplet triplet0;
            triplet0.anchor = scene_samples[obj][ran_sample];
            triplet0.puller = templates[obj][puller];
            triplet0.pusher = templates[obj][pusher];
            triplets.push_back(triplet0);


            // Store puller as new anchor for the next two triplets
            anchor = puller;


            /// Type 1: All templates are from same object but first two are closer than the third

            // Find puller template with closest pose
            puller=0;
            best_dist = numeric_limits<float>::max();
            for (size_t temp=0; temp < nr_poses; temp++)
            {
                if (temp == anchor) continue; // we skip the anchor template
                float temp_dist = tpl_quat[anchor].angularDistance(tpl_quat[temp]);
                if (temp_dist >= best_dist) continue;
                puller = temp;
                best_dist = temp_dist;
            }

            // Randomize through until pusher is neither anchor nor puller
            pusher = ran_tpl(ran);
            while ((pusher == anchor) || (pusher == puller)) pusher = ran_tpl(ran);

            Triplet triplet1;
            triplet1.anchor = templates[obj][anchor];
            triplet1.puller = templates[obj][puller];
            triplet1.pusher = templates[obj][pusher];
            triplets.push_back(triplet1);

            /// Type 2: two templates are from same object, the third from another

            // Randomize through until anchor != puller
            puller = ran_tpl(ran);
            while (puller == anchor) puller = ran_tpl(ran);

            // Randomize through until pusher is another object
            pusher = ran_obj(ran);
            while (pusher == obj) pusher = ran_obj(ran);

            Triplet triplet2;
            triplet2.anchor = templates[obj][anchor];
            triplet2.puller = templates[obj][puller];
            triplet2.pusher = templates[pusher][ran_tpl(ran)];
            triplets.push_back(triplet2);



#if 0      // Show triplets
            for (size_t idx = triplets.size()-3; idx < triplets.size(); idx++)
            {
                imshow("anchor",showRGBDPatch(triplets[idx].anchor.data,false));
                imshow("puller",showRGBDPatch(triplets[idx].puller.data,false));
                imshow("pusher",showRGBDPatch(triplets[idx].pusher.data,false));
                waitKey();
            }
#endif

            cout << "Done building triplets!" << endl;
        }

        // Check if we are finished (if all templates of all objects were anchors once)
        for (auto &vec : tmpl_used)
            for (int i=0; i < nr_poses; ++i) finished &= vec[i];

        finished = triplets.size()>100000;

    }


    // Since Caffe's HDF5Layer has a weird memory restriction, we chunk the training data into 1GB files
    // We also make sure that the chunk is divisible by 192 (i.e. a batch size that is fully divisible by 3)
    ofstream train(hdf5_path + "train_files.txt");
    vector<Sample> out;
    int chunk_counter=0;
    for (Triplet &tri : triplets)
    {
        out.push_back(tri.anchor);
        out.push_back(tri.puller);
        out.push_back(tri.pusher);

        int img_data = out[0].data.cols*out[0].data.rows*out[0].data.channels();
        int label_data = out[0].label.cols*out[0].label.rows*out[0].label.channels();
        int byte_size = out.size()*(img_data + label_data)*sizeof(float);
        if ((out.size() % 192 == 0) && (byte_size > 1024*1024*1024))
        {
            string filename = hdf5_path + "train_chunk"+to_string(chunk_counter)+".h5";
            writeHDF5(filename,out);
            train << filename << endl;
            out.clear();
            chunk_counter++;
        }
    }

    cout << "Done building triplets!" << endl;

}

TripletsPairs buildTripletsSergey(vector<string> used_models)
{
    vector<Triplet> triplets;
    vector<Pair> pairs;
    TripletsPairs triplets_pairs;

    size_t nr_objects = used_models.size();
    assert(nr_objects > 1);

    // Read stored templates and scene samples for the used models
    vector< vector<Sample> > training, templates, temp;
    for (string &seq : used_models)
    {
        training.push_back(readHDF5(hdf5_path + "realSamples_" + seq +".h5"));
        temp.push_back(readHDF5(hdf5_path + "synthSamples_" + seq +".h5"));
        training[training.size()-1].insert(training[training.size()-1].end(), temp[0].begin(), temp[0].end());
        temp.clear();
        templates.push_back(readHDF5(hdf5_path + "templates_" + seq + ".h5"));


    }

//    // Shuffle the vectors
//    for  (int i=0; i < nr_objects; ++i)
//    {
//        std::random_shuffle(training[i].begin(),training[i].end());
//        std::random_shuffle(templates[i].begin(),templates[i].end());
//    }

    // Read quaternion poses from training data
    vector< vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > > training_quats(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
    {
        training_quats[i].resize(training[i].size());
        for (size_t k=0; k < training_quats[i].size(); ++k)
            for (int j=0; j < 4; ++j)
                training_quats[i][k].coeffs()(j) = training[i][k].label.at<float>(0,1+j);
    }

    // Read quaternion poses from templates (they are identical for all objects)
    int nr_poses = templates[0].size();
    vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > tmpl_quats(nr_poses);
    for  (int i=0; i < nr_poses; ++i)
        for (int j=0; j < 4; ++j)
            tmpl_quats[i].coeffs()(j) = templates[0][i].label.at<float>(0,1+j);

    // Build a bool vector for each object that stores if all templates have been used yet
    vector< vector<bool> > tmpl_used(nr_objects);
    for (auto &vec : tmpl_used)
        vec.assign(nr_poses,false);

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_poses-1);

    // Random generators that returns a random scene sample for a given object
    vector< std::uniform_int_distribution<size_t> > ran_training_sample(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
        ran_training_sample[i] = std::uniform_int_distribution<size_t>(0, training[i].size()-1);


    for(bool finished=false; !finished;)
    {
        size_t anchor, puller, pusher, oldpusher;

        // Build each triplet by cycling through each object
        for (size_t obj=0; obj < nr_objects; obj++)
        {

            /// Type 0: A random scene sample together with closest template against another template

            // Pull random scene sample and find closest pose neighbor from templates
            size_t ran_sample = ran_training_sample[obj](ran);
            puller=0;
            float best_dist = numeric_limits<float>::max();
            for (size_t temp=0; temp < nr_poses; temp++)
            {
//                if (tmpl_used[obj][temp]) continue;  // Skip if template already used
                float temp_dist = training_quats[obj][ran_sample].angularDistance(tmpl_quats[temp]);
                if (temp_dist >= best_dist) continue;
                puller = temp;
                best_dist = temp_dist;
            }


            // Mark template as used
//            tpml_used[obj][puller] = true;


            // Randomize through until pusher != puller
            pusher = ran_tpl(ran);
            while (pusher == puller) pusher = ran_tpl(ran);

            Triplet triplet0;
            triplet0.anchor = training[obj][ran_sample];
            triplet0.puller = templates[obj][puller];
            triplet0.pusher = templates[obj][pusher];
            triplets.push_back(triplet0);
            oldpusher = pusher;


            /// Type 1: All templates are from same object but first two are closer than the third

            // Randomize through until pusher is neither anchor nor puller
            pusher = ran_tpl(ran);
            while ((pusher == puller) && (pusher == oldpusher)) pusher = ran_tpl(ran);

            Triplet triplet1;
            triplet1.anchor = training[obj][ran_sample];
            triplet1.puller = templates[obj][puller];
            triplet1.pusher = templates[obj][pusher];
            triplets.push_back(triplet1);

            /// Type 2: two templates are from same object, the third from another

            // Randomize through until pusher is another object
            pusher = ran_obj(ran);
            while (pusher == obj) pusher = ran_obj(ran);

            Triplet triplet2;
            triplet2.anchor = training[obj][ran_sample];
            triplet2.puller = templates[obj][puller];
            triplet2.pusher = templates[pusher][ran_tpl(ran)];
            triplets.push_back(triplet2);


#if 0      // Show triplets
            for (size_t idx = triplets.size()-3; idx < triplets.size(); idx++)
            {
                imshow("anchor",showRGBDPatch(triplets[idx].anchor.data,false));
                imshow("puller",showRGBDPatch(triplets[idx].puller.data,false));
                imshow("pusher",showRGBDPatch(triplets[idx].pusher.data,false));
                waitKey();
            }

#endif

            /// Add pairs
            Pair pair;
            pair.anchor = training[obj][ran_sample];
            pair.puller = templates[obj][puller];
            pairs.push_back(pair);

        }

        // Check if we are finished (if all templates of all objects were anchors once)
        for (auto &vec : tmpl_used)
            for (int i=0; i < nr_poses; ++i) finished &= vec[i];

        finished = triplets.size()>100000;

    }
    triplets_pairs.triplets = triplets;
    triplets_pairs.pairs = pairs;

    return triplets_pairs;

}


void trainNet()
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);

    caffe::SolverParameter solver_param;
    solver_param.set_base_lr(0.0001);
    solver_param.set_momentum(0.9);
    solver_param.set_weight_decay(0.0005);

    solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);

    solver_param.set_stepsize(1000);
    solver_param.set_lr_policy("step");
    solver_param.set_gamma(0.9);

    solver_param.set_max_iter(150000);
    //solver_param.set_test_interval(100);
    //solver_param.add_test_iter(1);

    solver_param.set_snapshot(25000);
    solver_param.set_snapshot_prefix("manifold");

    solver_param.set_display(1);
    solver_param.set_net(network_path + "manifold_train.prototxt");
    caffe::SGDSolver<float> *solver = new caffe::SGDSolver<float>(solver_param);
    string resume = network_path + "manifold_iter_50000.solverstate";
    //    solver->Solve(resume);
    solver->Solve();
}

void trainNetOnline(vector<string> used_models)
{
    caffe::SolverParameter solver_param;
    solver_param.set_base_lr(0.0001);
    solver_param.set_momentum(0.9);
    solver_param.set_weight_decay(0.0005);

    solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);

    solver_param.set_stepsize(1000);
    solver_param.set_lr_policy("step");
    solver_param.set_gamma(0.9);

    int max_iters = 150000;
    solver_param.set_max_iter(150000);

    solver_param.set_snapshot(25000);
    solver_param.set_snapshot_prefix("manifold");

    solver_param.set_display(1);
    solver_param.set_net(network_path + "manifold_train.prototxt");
    caffe::SGDSolver<float> *solver = new caffe::SGDSolver<float>(solver_param);
////    solver->Restore(network_path + "manifold_iter_50000.solverstate");

    // Read the scene samples
    vector<Sample> batch;
    vector<Triplet> triplets;
    vector<Pair> pairs;
    TripletsPairs triplets_pairs;
    triplets_pairs = buildTripletsSergey(used_models);

    // Get network information
    boost::shared_ptr<caffe::Net<float> > net = solver->net();
    caffe::Blob<float>* input_data_layer = net->input_blobs()[0];
    caffe::Blob<float>* input_label_layer = net->input_blobs()[1];
    const size_t batchSize = input_data_layer->num();
    const int channels =  input_data_layer->channels();
    const int targetSize = input_data_layer->height();
    const int slice = input_data_layer->height()*input_data_layer->width();
    const int img_size = slice*channels;

    vector<float> data(batchSize*img_size,0), labels(batchSize,0);

    // Perform training
    for (int iter = 0; iter <= max_iters; iter++)
    {
        // Fill current batch
        int batch_pairs = 132;
        int batch_triplets = 198;
        batch.clear();

        // Loop through the number of samples per batch
        for (int sampleId = iter*batchSize; sampleId < iter*batchSize; ++sampleId)
        {
            // - fill triplets [:198]
            for (int tripletId = (sampleId-(batch_pairs*iter))/3; tripletId < (sampleId-(batch_pairs*iter))/3 + batch_triplets/3; ++tripletId) {
                batch.push_back(triplets_pairs.triplets[tripletId].anchor);
                batch.push_back(triplets_pairs.triplets[tripletId].puller);
                batch.push_back(triplets_pairs.triplets[tripletId].pusher);
            }
            // - fill pairs [198:330]
            for (int pairId = (sampleId-(batch_triplets*iter))/2; pairId < (sampleId-(batch_triplets*iter))/2 + batch_pairs/2; ++pairId) {
                batch.push_back(triplets_pairs.pairs[pairId].anchor);
                batch.push_back(triplets_pairs.pairs[pairId].puller);
            }
        }

        // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
        for (size_t i=0; i < batch.size(); ++i)
        {
            int currImg = i*img_size;
            for (int ch=0; ch < channels ; ++ch)
                for (int y = 0; y < targetSize; ++y)
                    for (int x = 0; x < targetSize; ++x)
                        data[currImg + slice*ch + y*targetSize + x] = batch[i].data.ptr<float>(y)[x*channels + ch];
            labels[i] = batch[i].label.at<float>(0,0);
        }
        input_data_layer->set_cpu_data(data.data());
        input_label_layer->set_cpu_data(labels.data());
        solver->Step(1);
    }
}


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
            vector< caffe::Blob<float>* > out = CNN.ForwardPrefilled();

            for (size_t j=0; j < currIdxs.size(); ++j)
                memcpy(descs.ptr<float>(currIdxs[j]), out[0]->cpu_data() + j*desc_dim, desc_dim*sizeof(float));

            currIdxs.clear(); // Throw away current batch
        }
    }

    return descs;

}

void testNet()
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    caffe::Net<float> CNN(network_path + "manifold_test.prototxt", caffe::TEST);
    CNN.CopyTrainedLayersFrom(network_path + "manifold_iter_25000.caffemodel");
    
    vector<Sample> ape = readHDF5(hdf5_path + "templates_ape.h5");
    ape.resize(301);
    Mat ape_descs = computeDescriptors(CNN,ape);


    vector<Sample> driller = readHDF5(hdf5_path + "templates_driller.h5");
    driller.resize(301);
    Mat driller_descs = computeDescriptors(CNN,driller);

    Mat descs;
    descs.push_back(ape_descs);
    descs.push_back(driller_descs);

    cerr << descs << endl;

    // Visualize for the case where feat_dim is 3D
    viz::Viz3d lol;
    cv::Mat vizMat(descs.rows,1,CV_32FC3, descs.data);
    cv::Mat colorMat;
    colorMat.push_back(Mat(301,1,CV_8UC3,Scalar(255,0,0)));
    colorMat.push_back(Mat(301,1,CV_8UC3,Scalar(0,255,0)));

    lol.showWidget("cloud",viz::WCloud(vizMat,colorMat));
    //lol.setViewerPose(cv::Affine3f::Identity());
    lol.spin();
    
}

void testKNN(bool realData)
{
    // Model name to check
    string seq = "ape";

    // Camera matrix
    Matrix3f cam = Matrix3f::Identity();
    cam(0,0) = 572.4114f;
    cam(0,2) = 325.2611f;
    cam(1,1) = 573.5704f;
    cam(1,2) = 242.0489f;

    // Load the snapshot
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    caffe::Net<float> CNN(network_path + "manifold_test.prototxt", caffe::TEST);
    CNN.CopyTrainedLayersFrom(network_path + "manifold_iter_25000.caffemodel");

    // Read the templates, compute the descriptors and store them to DB
    vector<Sample> ape = readHDF5(hdf5_path + "templates_ape.h5");
    ape.resize(301);
    Mat DBfeats = computeDescriptors(CNN,ape);

    // Load model
    Model model;
    model.loadPLY(LINEMOD_path + seq + ".ply");

    // Load sceneSamples
    vector<Sample> sceneSamples = readHDF5(hdf5_path + "scenesamples_" + seq + ".h5");

    // Create the renderer
    SphereRenderer renderer;
    renderer.init(cam);
    Mat col,dep;
    Isometry3f pose = Isometry3f::Identity();
    pose.translation()(2) = 0.4f;

    // Randomize the angle
    std::uniform_real_distribution<float> angle(0, M_PI/2);
    std::uniform_real_distribution<float> a(0, 2*M_PI), i(0, M_PI/2);

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    matcher->add(DBfeats);

    // Enter the infinite loop
    while(true)
    {
        // Test on random synthetic data
        if(!realData)
        {
            // - use the half sphere with random azi and inc to render
            float azi = a(ran);
            float inc = i(ran);
            Vector3f vertex(sin(inc)*cos(azi),sin(inc)*sin(azi),cos(inc));
            pose.linear() = renderer.computeRotation(vertex);

            // - assign random angles to the poses
            //Matrix3f randomAngles = AngleAxisf(angle(ran), Vector3f::UnitX())
            //                      * AngleAxisf(angle(ran), Vector3f::UnitY())
            //                      * AngleAxisf(angle(ran), Vector3f::UnitZ()).matrix();
            //pose.linear() = randomAngles;

            renderer.renderView(model, pose, col, dep, false);

            // - generate the sample
            vector<Sample> samples;
            Sample sample;
            sample.data = samplePatchWithScale(col, dep, cam(0,2), cam(1,2), pose.translation()(2), cam(0,0), cam(1,1));
            samples.push_back(sample);

            // - compute the descriptor of the new sample
            Mat newDescs = computeDescriptors(CNN,samples);

            // - match the sample
            vector< vector<DMatch> > matches;
            matcher->knnMatch(newDescs, matches, 5);

            // - show the result
            imshow("template",showRGBDPatch(sample.data,false));

            for (DMatch &m : matches[0])
            {
                // - show the result
                imshow("kNN",showRGBDPatch(samples[0].data,false));
                imshow("kNN",showRGBDPatch(ape[m.trainIdx].data,false));

//                    Quaternionf sampleQuat, kNNQuat;
//                    for (int i=0; i < 4; ++i)
//                    {
//                        sampleQuat.coeffs()(i) = samples[0].label.at<float>(0,1+i);
//                        kNNQuat.coeffs()(i) = ape[m.trainIdx].label.at<float>(0,1+i);
//                    }


                // -- compute angle difference
//                cout << "Angle difference: " << sampleQuat.angularDistance(kNNQuat) << endl;

                waitKey();
            }

            imshow("kNN",showRGBDPatch(ape[matches[0][0].trainIdx].data,false));
            waitKey();

        // Test on real data
        } else {

            std::uniform_int_distribution<size_t> sampleId(0, sceneSamples.size()-1);


            // - take random sample
            vector<Sample> samples;
            samples.push_back(sceneSamples[sampleId(ran)]);

            // - compute the descriptor of the new sample
            Mat newDescs = computeDescriptors(CNN,samples);

            // - match the sample
            vector< vector<DMatch> > matches;
            int knn=5;
            matcher->knnMatch(newDescs, matches, knn);

            // - show the result
            imshow("sceneSample",showRGBDPatch(samples[0].data,false));

            for (DMatch &m : matches[0])
            {
                    imshow("kNN",showRGBDPatch(ape[m.trainIdx].data,false));

                    Quaternionf sampleQuat, kNNQuat;
                    for (int i=0; i < 4; ++i)
                    {
                        sampleQuat.coeffs()(i) = samples[0].label.at<float>(0,1+i);
                        kNNQuat.coeffs()(i) = ape[m.trainIdx].label.at<float>(0,1+i);
                    }

                // -- compute angle difference
                cout << "Angle difference: " << sampleQuat.angularDistance(kNNQuat) << endl;

                waitKey();
            }

        }

    }

}


int main(int argc, char *argv[])
{
    // For each object of the dataset
    for (size_t i = 0; i < models.size(); ++i)
    {
        // Build mapping from model name to index number
        model_index[models[i]] = i;

        // Create all samples for this object (needs to be done only once)
        createSceneSamplesAndTemplates(models[i]);
    }

    // buildTriplets(vector<string>({"ape","bowl","cam"}));
    // trainNet();

    trainNetOnline(vector<string>({"ape","bowl","cam"}));

    // testNet();
    // testKNN(1);


    return 0;
}






