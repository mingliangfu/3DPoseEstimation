
#pragma once

#include <chrono>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>



using namespace cv;
using namespace Eigen;
using namespace std;


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



