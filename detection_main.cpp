

#include <fstream>
#include <sstream>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/features2d.hpp>

#include <tbb/tbb.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "caffe/caffe.hpp"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "include/model.h"
#include "include/hdf5handler.h"
#include "include/datasetmanager.h"
#include "include/networkevaluator.h"
#include "include/networksolver.h"
#include "include/utilities.h"
#include "include/icp.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace boost;


#define SQR(x) ((x)*(x))

Matrix3f cam;
hdf5Handler h5;
std::default_random_engine rand_eng;


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


void loadParams(string linemod_path)
{
    assert(filesystem::exists(linemod_path+"diameters.txt"));
    ifstream dia(linemod_path+"diameters.txt");
    for (int i=0; i < 15; i++)
    {
        pair<string,float> temp;
        dia >> temp.first >> temp.second;
        diameters[temp.first] = temp.second;
    }

    diameters["ape"] *= 1.3f;
}


// Takes a color and depth frame and samples a normalized 4-channel patch at the given center position and z-scale
Mat samplePatchWithScale(Mat &color, Mat &depth, Mat &normals, int center_x, int center_y, float z, float fx, float fy)
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
    Mat temp_col, temp_dep, temp_nor, temp, final;
    color(cut).copyTo(temp_col);
    depth(cut).copyTo(temp_dep);
    normals(cut).copyTo(temp_nor);

    // Convert to float
    temp_col.convertTo(temp_col,CV_32FC3,1/255.f);

    // Demean with central z value, clamp and rescale to [0,1]
    temp_dep -= z;
    temp_dep.setTo(-m, temp_dep < -m);
    temp_dep.setTo(m, temp_dep > m);
    temp_dep *= 1.0f / m;
    temp_dep = (temp_dep+1.f)*0.5f;

    // Resize
    const int CNN_INPUT_SIZE = 64;
    const Size final_size(CNN_INPUT_SIZE,CNN_INPUT_SIZE);
    resize(temp_col,temp_col,final_size);   // Standard bilinear interpolation
    resize(temp_nor,temp_nor,final_size);   // Standard bilinear interpolation
    resize(temp_dep,temp_dep,final_size,0,0,INTER_NEAREST);// Nearest-neighbor interpolation for depth!!!

    cv::merge(vector<Mat>{temp_col,temp_dep,temp_nor},final);



    return final;
}



struct View
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Quaternionf rot;
    vector<Point> points;
    vector<float> depths;
};

struct Hypo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Isometry3f pose;
    Mat col, dep, nor;
    pair<int,int> offset;
    vector<Point> points;
    vector<Vector3f, Eigen::aligned_allocator<Vector3f> > cloud_pts;
    View *NN;
    IcpStruct icp_ret;
    Point pos2D;
    float depth_sim;
    int NNIdx, sampleIdx;
    float distance;
    bool killed;
    Hypo() : depth_sim(0), killed(false), pose(Isometry3f::Identity()){}
};



struct SceneSample
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point pos2D;
    Vector3f pos3D;
    Mat patch, feat;
};

vector<SceneSample, Eigen::aligned_allocator<SceneSample> > sampleScene(Mat &col, Mat &dep, Mat &nor, int step_size)
{
    int y_samples = (dep.rows / step_size)-1;
    int x_samples = (dep.cols / step_size)-1;

#if 0
    Mat temp;
    f.color.copyTo(temp);
    for (SceneSample &s : scene_samples)
    {
        if (s.patch.empty()) cv::circle(temp,s.pos2D,2,Scalar(0,0,255));
        else cv::circle(temp,s.pos2D,2,Scalar(0,255,0));
    }
    imshow("samples",temp); waitKey();
#endif


    // Collect samples
    vector<SceneSample, Eigen::aligned_allocator<SceneSample> > out;
    const int border = 10;
    for (int y=border; y <= y_samples-border; y++)
        for (int x=border; x <= x_samples-border; x++)
        {
            int gy = y*step_size, gx = x*step_size;
            if (y%2==0) gx += step_size/2;  // Add x-offset every odd row
            float d = dep.at<float>(gy,gx);
            if (d==0) continue;
            SceneSample s;
            s.patch = samplePatchWithScale(col,dep,nor,gx,gy,d,cam(0,0),cam(1,1));
            if (s.patch.empty()) continue;
            s.pos2D.x = gx;
            s.pos2D.y = gy;
            out.push_back(s);
        }
    return out;
}

void binarizeDescriptors(Mat &descs)
{
    Mat binDescs = Mat::zeros(descs.rows,descs.cols/8,CV_8U);
    for (int r=0; r < descs.rows; ++r)
        for (int b=0; b < descs.cols; ++b)
        {
            int curr_byte = b/8;
            int curr_bit = b - (curr_byte*8);
            binDescs.at<uchar>(r,curr_byte) |= (descs.at<float>(r,b) >= 0) << curr_bit;
        }
    binDescs.copyTo(descs);
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

        depth2normals(f.depth,f.normals,cam);

        Sample s;
        s.data = samplePatchWithScale(f.color,f.depth,f.normals, projCentroid(0),projCentroid(1),z,cam(0,0),cam(1,1));

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




map<int, vector<View, Eigen::aligned_allocator<View> > > tpl_views;

void createTemplatePoints(Model &model,Matrix3f &cam)
{

    // Create synthetic views
    SphereRenderer sphere(cam);
    Vector3f scales(0.6, 0.1, 1.11);     // Render from 0.4 meters
    Vector3f in_plane_rots(-45,15,45);  // Render in_plane_rotations from -45 degree to 45 degree in 15degree steps
    vector<RenderView, Eigen::aligned_allocator<RenderView> > views =
            sphere.createViews(model,2,scales,in_plane_rots,true,false,false);    // Equidistant sphere sampling with recursive level subdiv

    for (RenderView &v : views)
    {
        int scale = (v.pose.translation()(2)+0.05f)*10;
        View view;
        Mat fg = v.dep > 0;
        float inc = cv::countNonZero(fg)/1000.f;
        for (float x=0; x < v.dep.cols; x += inc)
            for (float y=0; y < v.dep.rows; y += inc)
                if(fg.at<uchar>(y,x))
                {
                    view.depths.push_back(v.dep.at<float>(y,x));
                    view.points.push_back(Point(x-cam(0,2),y - cam(1,2)));
                }
        view.rot = Quaternionf(v.pose.linear());
        tpl_views[scale].push_back(view);
    }

    cerr << "Scales: " << endl;
    for (auto &k : tpl_views) cerr << k.first << endl;

}

void randomColorFill(Mat &patch)
{
    int chans = patch.channels();
    bool nors = patch.channels() == 7;
    std::uniform_real_distribution<float> p(0.f,1.f);
    std::uniform_real_distribution<float> p_nor(-1.f,1.f);
    for (int r=0; r < patch.rows; ++r)
        for (int c=0; c < patch.cols; ++c)
        {
            float *row = patch.ptr<float>(r);
            if (row[c*chans + 3] > 0) continue;
            for (int ch=0; ch < 3; ++ch)
                row[c*chans + ch] =  p(rand_eng);
            if (nors)
                for (int ch=4; ch < 7; ++ch)
                    row[c*chans + ch] =  p_nor(rand_eng);
        }
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
        //z = v.pose.translation()(2);

        Mat normals;
        depth2normals(v.dep,normals,cam);

        Sample sample;
        sample.data = samplePatchWithScale(v.col,v.dep,normals,cam(0,2),cam(1,2),z,cam(0,0),cam(1,1));
        randomColorFill(sample.data);

        // Build 6-dimensional label: model index + quaternion + z-offset
        sample.label = Mat(1,6,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(v.pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);
        sample.label.at<float>(0,5) = v.pose.translation()(2)-z;

        samples.push_back(sample);
    }
    return samples;

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



void createSceneSamplesAndTemplates(datasetManager &ds)
{

    for (size_t modelId = 0; modelId < ds.models.size(); ++modelId)
    {

        string model_name = ds.models[modelId];

        clog << "\nCreating samples and patches for " << model_name << ":" << endl;

        // - load model
        Model model;
        model.loadFile(ds.dataset_path + model_name + ".ply");

        // - load frames of benchmark and visualize
        Benchmark bench = loadLinemodBenchmark(ds.dataset_path, model_name);
        cam = bench.cam;

        // === Real data ===
        // - for each scene frame, extract RGBD sample
        vector<Sample> realSamples = extractSceneSamples(bench.frames,bench.cam,modelId);

        // - shuffle the samples
        random_shuffle(realSamples.begin(), realSamples.end());

        // - store realSamples to HDF5 files
        ds.h5.write(ds.hdf5_path + "realSamples_" + model_name +".h5", realSamples);
        //for (Sample &s : realSamples) showRGBDPatch(s.data);

        // === Synthetic data ===
        clog << "  - render synthetic data:" << endl;
        // - create synthetic samples and templates
        int subdivTmpl = 2; // sphere subdivision factor for templates
        vector<Sample> templates = createTemplates(model,bench.cam,modelId, subdivTmpl);
        vector<Sample> synthSamples = createTemplates(model,bench.cam,modelId, subdivTmpl+1);

        // - store realSamples to HDF5 files
        ds.h5.write(ds.hdf5_path + "templates_" + model_name + ".h5", templates);
        ds.h5.write(ds.hdf5_path + "synthSamples_" + model_name + ".h5", synthSamples);
        //for (Sample &s : templates) showRGBDPatch(s.data);

    }
}


void renderExtract(vector<Hypo, Eigen::aligned_allocator<Hypo> > &hypos, Mat &cloud, SphereRenderer &renderer,Model &m)
{
    for (Hypo &h : hypos) h.offset = renderer.renderView(m,h.pose,h.col,h.dep);
    tbb::parallel_for<size_t>(0, hypos.size(), [&] (size_t i)
    {
        Hypo &h = hypos[i];
        h.points.clear();
        for (int r=0; r < h.dep.rows;++r)
            for (int c=0; c < h.dep.cols;++c)
                if (h.dep.at<float>(r,c))
                {
                    Point p(c+h.offset.first,r+h.offset.second);
                    if((p.x <0) || (p.y <0) || (p.x > cloud.cols-1) || (p.y > cloud.rows-1)) continue;
                    h.points.push_back(p);
                }

        h.cloud_pts.clear();
        for (Point &p : h.points) h.cloud_pts.push_back(cloud.at<Vector3f>(p));
    });
}



int main(int argc, char *argv[])
{

    Model m;
    m.loadFile("optimized_tsdf_texture_mapped_mesh.obj");


    if (!m.m_texture.empty())
    {
        imshow("tex",m.m_texture);
        for (Vec2f t : m.m_tcoords) cerr << t << endl;
        waitKey();
    }

    string config_file = "manifold_rgbnor_16.ini";

    datasetManager dm("configs/" + config_file);
    dm.generateDatasets();
    networkSolver solver("configs/" + config_file,&dm);

    //createSceneSamplesAndTemplates(dm);


    loadParams(dm.dataset_path);

    bool binary = solver.binarization;

#if 0
    if (binary) solver.binarizeNet(33000);
    else solver.trainNet(0);
#endif


    caffe::Net<float> *net;
    if (binary)
    {
        net = new caffe::Net<float>(solver.network_path + solver.binarization_net_name + ".prototxt",caffe::TEST);
        net->CopyTrainedLayersFrom(solver.binarization_net_name + "_iter_" + to_string(5500) + ".caffemodel");
    }
    else
    {
        net = new caffe::Net<float>(solver.network_path + solver.net_name + ".prototxt",caffe::TEST);
        net->CopyTrainedLayersFrom(solver.net_name + "_iter_" + to_string(33000) + ".caffemodel");
    }
    string seq = "can";

    Model model;
    model.loadFile(dm.dataset_path + seq + ".ply");
    ICP icp;
    bool icp_dump = icp.load3DTransform(seq+".icp");
    icp.setData(model);
    if (!icp_dump) icp.dump3DTransform(seq+".icp");


    Benchmark bench = loadLinemodBenchmark(dm.dataset_path,seq);
    cam = bench.cam;

    vector<Sample> tpls = createTemplates(model,cam,0,2);


    Mat DB = networkEvaluator::computeDescriptors(*net,tpls);
    if (binary) binarizeDescriptors(DB);

    cv::Ptr<DescriptorMatcher> matcher;
    if (binary) matcher = DescriptorMatcher::create("BruteForce-Hamming");
    else matcher = DescriptorMatcher::create("BruteForce");
    matcher->add(DB);


    map<string, float> rgbdnor_thresholds = {
        {"iron",0.5f},
        {"benchvise",0.65f}
    };

    map<string, float> rgbdnor_bin_thresholds = {
        {"cat",45}
        };

    float thresh = rgbdnor_bin_thresholds[seq];
    SphereRenderer renderer(cam);

    createTemplatePoints(model,cam);

    float best_thresh = 0.f;
    float hit_count = 0;
    float correct_count = 0;
    for (Frame &f :  bench.frames)
    {

        depth2cloud(f.depth,f.cloud,cam);
        depth2normals(f.depth,f.normals,cam);

        cerr << "Frame " << to_string(f.nr) << ": ";

        vector<SceneSample, Eigen::aligned_allocator<SceneSample> > scene_samples = sampleScene(f.color,f.depth,f.normals,8);

        vector<Sample> patches;
        for (auto &s : scene_samples) {Sample tmp; tmp.data = s.patch; patches.push_back(tmp);};

        Mat queries = networkEvaluator::computeDescriptors(*net,patches);
        if (binary) binarizeDescriptors(queries);

        vector< vector<DMatch> > matches;
        int knn = 1;
        matcher->knnMatch(queries, matches, knn);

        vector<Hypo, Eigen::aligned_allocator<Hypo> > hypos;
        for (size_t sample=0; sample < scene_samples.size(); sample++)
        {
            for (DMatch &m : matches[sample])
            {
               // if (m.distance > thresh) continue;
                Hypo h;
                h.NNIdx = m.trainIdx;
                h.sampleIdx = m.queryIdx;
                h.pos2D = scene_samples[sample].pos2D;
                h.distance = m.distance;
                Quaternionf q;
                for (int i=0; i < 4; ++i) q.coeffs()(i) = tpls[h.NNIdx].label.at<float>(0,1+i);
                h.pose.linear() = q.toRotationMatrix();
                h.pose.translation() = f.cloud.at<Vector3f>(h.pos2D);
                h.pose.translation()(2) += tpls[h.NNIdx].label.at<float>(0,5);
                hypos.push_back(h);
            }
        }

        cerr << "#Hypos: " << hypos.size() << endl;


        f.depth.setTo(0, f.depth>1.2f);
        depth2cloud(f.depth,f.cloud,cam);

        tbb::parallel_for<size_t>(0, hypos.size(), [&] (size_t i)
        {
            Hypo &h = hypos[i];
            int scale = (h.pose.translation()(2)+0.05f)*10;
            if (scale < 6 || scale > 11) return;
            Quaternionf rot(h.pose.linear());
            h.NN = nullptr;
            float best_dist = numeric_limits<float>::max();
            for (View &v : tpl_views[scale])
            {
                float dist = rot.angularDistance(v.rot);
                if (dist>best_dist) continue;
                best_dist = dist;
                h.NN = &v;
            }
            assert (h.NN != nullptr);

            h.cloud_pts.clear();
            for (size_t i=0; i < h.NN->points.size(); ++i)
            {
                Point g = h.NN->points[i]+h.pos2D;
                if((g.x <0) || (g.y <0) || (g.x > f.cloud.cols-1) || (g.y > f.cloud.rows-1)) continue;
                h.cloud_pts.push_back(f.cloud.at<Vector3f>(g));
            }
        });



        for (int refines = 0; refines < 1; refines++)
        {
            for (Hypo &h : hypos)
            {

                // Bring scene cloud from camera into local coordinates, run full model-based ICP and compute the updated pose
                Isometry3f pose_inv = h.pose.inverse();
                for(Vector3f &p : h.cloud_pts) p = pose_inv*p;
                h.icp_ret = icp.compute(h.cloud_pts,10,0.0001f);
                h.pose = h.pose*h.icp_ret.pose.inverse();
            }

            renderExtract(hypos, f.cloud, renderer,model);
        }

#if 1
        tbb::parallel_for<size_t>(0, hypos.size(), [&] (size_t i)
        {
            Hypo &h = hypos[i];
            h.killed = (h.icp_ret.error > 0.01f);
            if (h.killed) return;
            for (size_t i=0; i < h.points.size(); ++i)
            {
                Point &p = h.points[i];
                if((p.x <0) || (p.y <0) || (p.x > f.cloud.cols-1) || (p.y > f.cloud.rows-1)) continue;
                if (SQR(h.dep.at<float>(p.y-h.offset.second,p.x-h.offset.first)-f.depth.at<float>(p))<SQR(0.02f)) h.depth_sim++;
            }
            h.depth_sim /= h.points.size();
            h.killed = (h.depth_sim < 0.8f);
        });
#endif


#if 0
        for (Hypo &h : hypos)
        {
            if (h.killed) continue;
            imshow("query",showRGBDPatch(scene_samples[h.sampleIdx].patch,false));
            imshow("kNN",showRGBDPatch(tpls[h.NNIdx].data,false));
            cerr << h.distance << " \t " << h.depth_sim << " \t " << h.icp_ret.error << endl;
            Mat out1, out2; f.color.copyTo(out1); f.color.copyTo(out2);
            //for (Point &p : h.NN->points)  cv::circle(out1,p+h.pos2D,1,Scalar(0,0,255));
            for (Point &p : h.points) cv::circle(out2,p,1,Scalar(0,255,0));imshow("out2",out2);
            waitKey();
        }
#endif



        bool found=false;
        float min_thresh = 200.f;
        for (Hypo &h : hypos)
        {
            if (h.killed) continue;
            if(linemod_error(f.gt,h.pose,model,seq)>0.1f*diameters[seq]) continue;
            found = true;
            cerr << h.distance << " \t " << h.depth_sim << " \t " << h.icp_ret.error << endl;
            min_thresh = std::min(min_thresh,h.distance);
        }

        if (found)
        {
            cerr << "FOUND" << endl;
            hit_count++;
            best_thresh   = std::max(best_thresh,min_thresh);
        }

        if (!hypos.empty())
        {
            Hypo &best_hypo = hypos[0];
            for (Hypo &h : hypos)
            {
                if (h.killed) continue;
                if (best_hypo.depth_sim < h.depth_sim)
                    best_hypo = h;
            }
            if(linemod_error(f.gt,best_hypo.pose,model,seq)<=0.1f*diameters[seq]) correct_count++;
        }

    }

    cerr << seq << endl;
    cerr << "Accuracy: " << correct_count/bench.frames.size() << endl;
    cerr << "Recall: " << hit_count/bench.frames.size() << endl;
    cerr << "Best thresh: " << best_thresh << endl;

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
