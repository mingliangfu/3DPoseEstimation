

#include "../include/utilities.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Eigenvalues>

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <random>
#include <set>

#include <H5Cpp.h>

#define SQR(a) ((a)*(a))



using namespace boost;
using namespace std;




void depth2cloud(const Mat &depth, Mat &cloud, float fx, float fy, float ox, float oy)
{
    assert(depth.type() == CV_32FC1);

    const float inv_fx = 1.0f/fx, inv_fy = 1.0f/fy;

    cloud = Mat::zeros(depth.size(),CV_32FC3);

    // Pre-compute some constants
    vector<float> x_cache(depth.cols);
    for (int x = 0; x < depth.cols; ++x)
        x_cache[x] = (x - ox) * inv_fx;

    for (int y = 0; y < depth.rows; ++y)
    {
        float val_y = (y - oy) * inv_fy;
        Vector3f *point = cloud.ptr<Vector3f>(y);
        const float* zs = depth.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x, ++point, ++zs)
        {
            float z = *zs;
            (*point) << x_cache[x] * z, val_y * z,z;
        }
    }
}

Mat maskPlaneRANSAC(Mat &cloud, Mat &normals)
{
    assert(!cloud.empty() && !normals.empty());
    default_random_engine gen;


    Vector3f *cloud_array = (Vector3f*) cloud.data;
    Vector3f *normal_array = (Vector3f*) normals.data;
    uniform_int_distribution<int> ran_index(0,cloud.cols*cloud.rows-1);

    vector<Vector3f> pts(3), nors(3);

    Vector3f best_plane;
    float best_off;
    int best_inliers=0;
    int iteration=0;
    while(true)
    {
        if (iteration>20) break;

        // Draw 3 unique points
        pts.clear();
        nors.clear();
        while(pts.size()<3)
        {
            int idx = ran_index(gen);
            if (cloud_array[idx](2)==0) continue;
            pts.push_back(cloud_array[idx]);
            nors.push_back(normal_array[idx]);
        }
        if (nors[0].dot(nors[1]) < 0.9) continue;
        if (nors[0].dot(nors[2]) < 0.9) continue;

        iteration++;
        Vector3f plane = ((pts[1]-pts[0]).cross(pts[2]-pts[0])).normalized();
        float off = -plane.dot(pts[0]);

        int inliers=0;
        for (int idx=0; idx < cloud.cols*cloud.rows; ++idx)
            if (std::abs(plane.dot(cloud_array[idx])+off) < 0.01f)
                inliers++;

        if (inliers > best_inliers)
        {
            best_plane = plane;
            best_off = off;
            best_inliers = inliers;
        }

    }

    Mat mask = Mat::zeros(cloud.size(),CV_8U);
    for (int r=0; r < cloud.rows; ++r)
        for (int c=0; c < cloud.cols; ++c)
            if (std::abs(best_plane.dot(cloud.at<Vector3f>(r,c))+best_off) < 0.01f)
                mask.at<uchar>(r,c) = 255;
    return mask;
}

void colors2gradient(Mat &image, Mat &gradient)
{
    assert(image.channels()==3);
    Mat smoothed,sobel_3dx,sobel_3dy;
    GaussianBlur(image,smoothed,Size(7,7),0);
    smoothed.convertTo(smoothed,CV_32FC3);
    Sobel(smoothed,sobel_3dx,CV_32F,1,0,3);
    Sobel(smoothed,sobel_3dy,CV_32F,0,1,3);

    vector<Mat> dx,dy;
    split(sobel_3dx,dx);
    split(sobel_3dy,dy);
    gradient = Mat::zeros(image.size(),CV_32FC3);
    for(int r=0;r<image.rows;++r)
        for(int c=0;c<image.cols;++c)
        {
            float mag0 = SQR(dx[0].at<float>(r,c))+SQR(dy[0].at<float>(r,c));
            float mag1 = SQR(dx[1].at<float>(r,c))+SQR(dy[1].at<float>(r,c));
            float mag2 = SQR(dx[2].at<float>(r,c))+SQR(dy[2].at<float>(r,c));
            float x,y,end;
            if(mag0>=mag1&&mag0>=mag2)
            {
                x = dx[0].at<float>(r,c);
                y = dy[0].at<float>(r,c);
                end = mag0;
            }
            else if(mag1>=mag0&&mag1>=mag2)
            {
                x = dx[1].at<float>(r,c);
                y = dy[1].at<float>(r,c);
                end  = mag1;
            }
            else
            {
                x = dx[2].at<float>(r,c);
                y = dy[2].at<float>(r,c);
                end  = mag2;
            }
            if(end>100) gradient.at<Vector3f>(r,c) = Vector3f(x,y,sqrt(end));
        }
}

void cloud2normals(Mat &cloud, Mat &normals)
{
    const int n=5;
    normals = Mat(cloud.size(),CV_32FC3, Scalar(0,0,0) );
    Matrix3f M;
    for(int r=n;r<cloud.rows-n-1;++r)
        for(int c=n;c<cloud.cols-n-1;++c)
        {
            const Vector3f &pt = cloud.at<Vector3f>(r,c);
            if(pt(2)==0) continue;

            const float thresh = 0.08f*pt(2);
            M.setZero();
            for (int i=-n; i <= n; i+=n)
                for (int j=-n; j <= n; j+=n)
                {
                    Vector3f curr = cloud.at<Vector3f>(r+i,c+j)-pt;
                    if (fabs(curr(2)) > thresh) continue;
                    M += curr*curr.transpose();
                }

#if 1
            Vector3f &no = normals.at<Vector3f>(r,c);
            EigenSolver<Matrix3f> es(M);
            int i; es.eigenvalues().real().minCoeff(&i);
            no = es.eigenvectors().col(i).real();
#else
            JacobiSVD<Matrix3f> svd(M, ComputeFullU | ComputeFullV);
            no = svd.matrixV().col(2);
#endif
            if (no(2) > 0) no = -no;

        }
}

void normals2curvature(Mat &normals, Mat &curvature)
{
    const int n=5;
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity(), cov;
    curvature = Mat(normals.size(),CV_32F,Scalar(0));

    Vector3f xyz_centroid;
    for(int r=10;r<normals.rows-11;++r)
        for(int c=10;c<normals.cols-11;++c)
        {

            Vector3f n_idx = normals.at<Vector3f>(r,c);

            if(n_idx(2)==0) continue;

            Matrix3f M = I - n_idx * n_idx.transpose();    // projection matrix (into tangent plane)

            // Project normals into the tangent plane
            vector<Vector3f> projected_normals;
            xyz_centroid.setZero ();
            for (int i=-n; i <= n; i+=n)
                for (int j=-n; j <= n; j+=n)
                {
                    projected_normals.push_back(M * normals.at<Vector3f>(r+i,c+j));
                    xyz_centroid += projected_normals.back();
                }

            xyz_centroid /= projected_normals.size();

            cov.setZero();  // Build scatter matrix of demeaned projected normals
            for (Vector3f &n : projected_normals)
            {
                Vector3f demean = n - xyz_centroid;
                cov += demean*demean.transpose();
            }

            EigenSolver<Matrix3f> es(cov,false);
            int i; es.eigenvalues().real().maxCoeff(&i);
            curvature.at<float>(r,c) = es.eigenvalues().real()[i];// * projected_normals.size();
        }
}

int linemodRGB2Hue(Vector3f &val)
{

    float V=max(val[0],max(val[1],val[2]));
    float min1=min(val[0],min(val[1],val[2]));

    float H=0;
    if(V==val[2])      H=60*(0+(val[1]-val[0])/(V-min1));
    else if(V==val[1]) H=60*(2+(val[0]-val[2])/(V-min1));
    else if(V==val[0]) H=60*(4+(val[2]-val[1])/(V-min1));

    int h = ((int)H) %360;
    H = h<0 ? h+360 : h;

    float S = (V == 0) ? 0 : (V-min1)/V;

    const float t=0.12f;

    // Map black to blue and white to yellow
    if(S<t) H=60;
    if(V<t) H=240;

    return H;
}

int linemodRGB2Hue(Vec3b &in)
{
    Vector3f val(in[2]/255.0f,in[1]/255.0f,in[0]/255.0f);
    return linemodRGB2Hue(val);
}

void colors2hues(Mat &colors, Mat &hues)
{
    assert(colors.type() == CV_8UC3);
    hues = Mat(colors.size(),CV_32S);

    for (int r = 0; r < colors.rows; ++r)
    {
        Vec3b *rgb = colors.ptr<Vec3b>(r);
        int *hsv = hues.ptr<int>(r);
        for (int c = 0; c < colors.cols; c++)
        {
            *hsv = linemodRGB2Hue(*rgb);
            rgb++;
            hsv++;
        }
    }
}

Mat imagesc(Mat &in)
{
    Mat temp,out;
    double mmin,mmax;
    cv::minMaxIdx(in,&mmin,&mmax);
    in.convertTo(temp,CV_8UC1, 255 / (mmax-mmin), -mmin);
    applyColorMap(temp, out, COLORMAP_JET);
    out.convertTo(out,CV_32FC3, 1/255.f);
    return out;
}


void depth2normals(const Mat &dep, Mat &nor,float fx, float fy, float ox, float oy)
{

    auto accum = [] (float delta,float i,float j,float *A,float *b)
    {
        float f = std::abs(delta)<0.05f;
        float fi=f*i;
        float fj=f*j;
        A[0] += fi*i;
        A[1] += fi*j;
        A[3] += fj*j;
        b[0]  += fi*delta;
        b[1]  += fj*delta;
    };

    nor = Mat::zeros(dep.size(),CV_32FC3);
    const int N=3, stride = dep.step1();
    for(int r=N;r<dep.rows-N-1;++r)
    {
        float *depth_ptr =  ((float*) dep.ptr(r))+N;
        for(int c=N;c<dep.cols-N-1;++c)
        {
            float d = *depth_ptr;
            if(d>0)
            {
                Vector3f normal;
                float A[4] = {0,0,0,0},b[2] = {0,0};

                for (int i=-N; i <= N; i+=N )
                    for (int j=-N; j <= N; j+=N )
                        accum(depth_ptr[i + j*stride]-d,i,j,A,b);
#if 0
                // angle-stable version
                float det = A[0]*A[3] - A[1]*A[1];
                float ddx = ( A[3]*b[0] - A[1]*b[1]) / det;
                float ddy = (-A[1]*b[0] + A[0]*b[1]) / det;
                normal(0) = -ddx*(d + ddy)*fx;
                normal(1) = -ddy*(d + ddx)*fy;
                normal(2) = (d+ddx*(x-ox+1))*(d+ddy*(y-oy+1) -ddx*ddy*(ox-x)*(oy-y));
#else
                normal(0) = ( A[3]*b[0] - A[1]*b[1])*fx;
                normal(1) = (-A[1]*b[0] + A[0]*b[1])*fy;
                normal(2) = ( A[0]*A[3] - A[1]*A[1])*d;
#endif

                float sqnorm = normal.squaredNorm();
                if (sqnorm>0) nor.at<Vector3f>(r,c) = normal/std::sqrt(sqnorm);
            }
            ++depth_ptr;
        }
    }
}

void bilateralDepthFilter(Mat &src, Mat &dst){


    auto accum = [](float depth, ushort coef, float refDepth, float depthVariance, float& total, float& totalCoef){
        if (depth > 0){
            float bilateralCoef = std::exp(-SQR(refDepth - depth) / (2 * depthVariance));
            total += coef * depth * bilateralCoef;
            totalCoef += coef * bilateralCoef;
        }
    };

    assert(src.channels() == 1);

    dst = Mat(src.size(),CV_32F,Scalar(0));

    const float* srcData = src.ptr<float>(0);
    float* dstData = dst.ptr<float>(0);

    const int N = 2;
    float depthUncertaintyCoef = 0.0285f;
    for (int y = N; y < src.rows - N; y++){
        int rowIndex = y * src.cols;
        for (int x = N; x < src.cols - N; x++){

            int index = (rowIndex + x);

            float central = srcData[index];
            if (central==0) continue;

            float depthSigma = depthUncertaintyCoef * central * central * 2.f;
            float depthVariance = SQR(depthSigma);

            float d00 = srcData[index - src.cols - N];
            float d01 = srcData[index - src.cols    ];
            float d02 = srcData[index - src.cols + N];
            float d10 = srcData[index - N];
            float d12 = srcData[index + N];
            float d20 = srcData[index + src.cols - N];
            float d21 = srcData[index + src.cols    ];
            float d22 = srcData[index + src.cols + N];

            float total = 0;
            float totalCoef = 0;
            accum(d00, 1, central, depthVariance, total, totalCoef);
            accum(d01, 2, central, depthVariance, total, totalCoef);
            accum(d02, 1, central, depthVariance, total, totalCoef);
            accum(d10, 2, central, depthVariance, total, totalCoef);
            accum(central, 4, central, depthVariance, total, totalCoef);
            accum(d12, 2, central, depthVariance, total, totalCoef);
            accum(d20, 1, central, depthVariance, total, totalCoef);
            accum(d21, 2, central, depthVariance, total, totalCoef);
            accum(d22, 1, central, depthVariance, total, totalCoef);

            float smooth = total / totalCoef;
            dstData[index] = smooth;
        }
    }
}

Vector3f rgb2hsv(Vector3f &rgb)
{
    Vector3f hsv;
    float min = rgb.minCoeff(),max = rgb.maxCoeff();
    hsv(2) = max;
    if (hsv(2) == 0)
    {
        hsv(1) = 0.0;
        hsv(0) = numeric_limits<float>::quiet_NaN();
        return hsv;
    }
    float delta = max - min;
    hsv(1) = 255 * delta / hsv(2);
    if (hsv(1) == 0)        hsv(0) = numeric_limits<float>::quiet_NaN();
    else if (max == rgb(0)) hsv(0) =     0 + 43.f*(rgb(1)-rgb(2))/delta;
    else if (max == rgb(1)) hsv(0) =  85.f + 43.f*(rgb(2)-rgb(0))/delta;
    else                    hsv(0) = 171.f + 43.f*(rgb(0)-rgb(1))/delta;
    return hsv;
}


