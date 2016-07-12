
#pragma once

#include <chrono>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>



using namespace cv;
using namespace Eigen;
using namespace std;



/*
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
pcl::PointCloud<pcl::PointXYZ>::Ptr CV2PCL(Mat &cloud):
Mat PCL2CV(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

void loadPNGstoPCL(std::string color_file,std::string depth_file,int i)
{
    float fx = 572.41140f;
    float ox = 325.26110f;
    float fy = 573.57043f;
    float oy = 242.04899f;

    cv::Mat color = imread(color_file);
    cv::Mat depth = imread(depth_file,-1);
    depth.convertTo(depth,CV_32F,0.001f);
    depth.setTo(std::numeric_limits<float>::quiet_NaN(), depth==0);

    cv::Mat cloud;
    depth2cloud(depth,cloud, fx, fy, ox, oy);

    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.width = depth.cols;
    pcl_cloud.height = depth.rows;
    pcl_cloud.points.resize(pcl_cloud.width*pcl_cloud.height);

    for (int c = 0; c < depth.cols; ++c)
        for (int r = 0; r < depth.rows; ++r)
        {
            pcl::PointXYZRGB point;
            point.x = cloud.at<Eigen::Vector3f>(r,c)(0);
            point.y = cloud.at<Eigen::Vector3f>(r,c)(1);
            point.z = cloud.at<Eigen::Vector3f>(r,c)(2);

            cv::Vec3b col = color.at<cv::Vec3b>(r,c);
            uint32_t rgb = (static_cast<uint32_t>(col[2]) << 16 |
            static_cast<uint32_t>(col[1]) << 8 | static_cast<uint32_t>(col[0]));
            point.rgb = *reinterpret_cast<float*>(&rgb);

            pcl_cloud.at(c,r) = point;
        }

    stringstream ss;
    ss << i;
    pcl::io::savePCDFileBinary("frame" + ss.str() + ".pcd",pcl_cloud);
}
*/


/*
void fuseDatasetIntoCloud(string dir_string)
{
    float fx = 572.41140f;
    float ox = 325.26110f;
    float fy = 573.57043f;
    float oy = 242.04899f;

    filesystem::path dir(dir_string );
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return;
    }

    detectionTUM::Model m;

    int last=150;
    cout << "Loading frames in the range " << 0 << " - " << last << endl;
    for (int i=120; i <= last;i+=1190)
    {

        ostringstream ss;
        ss << i;
        Mat color = imread(dir_string + "/data/color"+ss.str()+".jpg");
        Mat depth = imread(dir_string + "/data/depth"+ss.str()+".png",-1);
        assert(!color.empty() && !depth.empty());

        color.convertTo(color,CV_32FC3,1/255.0f);
        cvtColor(color,color,COLOR_RGB2BGR);
        depth.convertTo(depth,CV_32F,0.001);

        Isometry3f pose = Isometry3f::Identity();
        ifstream rot(dir_string + "/data/pose"+ss.str()+".txt");
        assert(rot.is_open());
        for (int k=0; k < 4;k++)
            for(int l=0; l < 4;l++)
                rot >> pose.matrix()(k,l);
        pose = pose.inverse();

        cv::Mat cloud;
        depth2cloud(depth,cloud, fx, fy, ox, oy);
        for (int r=0; r < cloud.rows; ++r)
            for (int c=0; c < cloud.cols; ++c)
            {
                Vector3f &p = cloud.at<Vector3f>(r,c);
                if (p(2)<0.5 || p(2)>1.5) continue;
                m.getPoints().push_back(pose*p);
                m.getColors().push_back(color.at<Vector3f>(r,c));
            }
    }

    m.savePLY("test.ply");
}
*/


namespace Gopnik
{

using Instance = pair<string,Isometry3f>;
struct Frame
{
    int nr;
    Mat color, depth, cloud, mask, hues, normals;
    vector<Instance, Eigen::aligned_allocator<Instance>> gt;
};

struct Sample
{
    Mat data, label;
    void copySample(Sample sample)
    {
        sample.data.copyTo(this->data);
        this->label = sample.label;
    }
};

struct Benchmark
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vector<Frame> frames;
    vector< pair<string,string> > models;
    Matrix3f cam;
};

pair<Mat,Mat> loadPCLCloud(string filename);


Benchmark loadKoboBenchmark(string folder);
Benchmark loadTejaniBenchmark(string folder);
Benchmark loadTejaniBenchmarkCorrected(string folder);
Benchmark loadTLESSBenchmark(string folder);
Benchmark loadKinectBenchmark(string folder);
Benchmark loadWillowBenchmarkOLD(string folder);
Benchmark loadExportBenchmark(string dir_string, int count=-1);
Benchmark loadWillowBenchmark(string folder);
Benchmark loadJHUBenchmark(string folder);
Benchmark loadToyotaBenchmark(string dir_string, int count=-1);
Benchmark loadLinemodBenchmark(string dir_string, int count=-1, int start=0);
Benchmark loadLinemodOcclusionBenchmark(string dir_string, int count=-1);

void writeHDF5(string filename, vector<Sample> &samples);
vector<Sample> readHDF5(string filename, int start=0, int count=-1);

void writeHDF5TensorFlow(string filename, vector<Sample> &samples);
vector<Sample> readHDF5TensorFlow(string filename, int counter=-1);



struct StopWatch
{
    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point now(){return chrono::high_resolution_clock::now();}
    StopWatch() : start(now()){}
    inline float restart() {float el=elapsedMs(); start = now(); return el;}
    inline float elapsedMs() {return 0.001f*chrono::duration_cast<chrono::microseconds>(now()-start).count();}
};


Mat imagesc(Mat &in);

Mat maskPlaneRANSAC(Mat &cloud, Mat &normals);


// Depth -> Cloud
void     depth2cloud(const Mat &depth, Mat &cloud, float fx, float fy, float ox, float oy);
inline Vector3f depth2cloud(Point p, float d, float fx, float fy, float ox, float oy)
{return Vector3f(d*(p.x-ox)/fx,d*(p.y-oy)/fy,d);}


inline void depth2cloud(const Mat &depth, Mat &cloud, Matrix3f &cam)
{depth2cloud(depth,cloud,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}

inline Vector3f depth2cloud(Point p, float d, Matrix3f &cam)
{return depth2cloud(p,d,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}

// Cloud -> Depth
inline Point cloud2depth(Vector3f &p,float fx,float fy,float ox, float oy)
{return Point2i(ox + p(0)*fx/p(2), oy + p(1)*fy/p(2));}


inline Point cloud2depth(Vector3f &p, Matrix3f &cam)
{return cloud2depth(p,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}

inline Point project(Vector3f p, Matrix3f &cam)
{return cloud2depth(p,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}


// CPU auxiliary functions
void colors2gradient(Mat &image, Mat &gradient);
void depth2normals(const Mat &depth, Mat &normals, float fx, float fy, float ox, float oy);
void cloud2normals(Mat &cloud, Mat &normals);
void normals2curvature(Mat &normals, Mat &curvature);

inline void depth2normals(const Mat &depth, Mat &normals, Matrix3f &cam)
{depth2normals(depth,normals,cam(0,0),cam(1,1),cam(0,2),cam(1,2));}


void bilateralDepthFilter(Mat &src, Mat &dst);

void colors2hues(Mat &colors, Mat &hues);

int linemodRGB2Hue(Vector3f &val);
int linemodRGB2Hue(Vec3b &in);


Vector3f linemodRGB2HSV(Vec3b in);

Vector3f rgb2hsv(Vector3f &rgb);


}
