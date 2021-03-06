#include "../include/helper.h"

namespace sz {

Mat showRGBDPatch(const Mat &patch, bool show /*= true*/)
{
    vector<Mat> channels;
    cv::split(patch,channels);

    Mat RGB,D,NOR,out;
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);

    if (channels.size()==4)
    {
        out = Mat(patch.rows,patch.cols*2,CV_32FC3);
        RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));
        D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));
    }
    else if (channels.size()==7)
    {
        out = Mat(patch.rows,patch.cols*3,CV_32FC3);
        cv::merge(vector<Mat>({abs(channels[4]),abs(channels[5]),abs(channels[6])}),NOR);
        RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));
        D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));
        NOR.copyTo(out(Rect(patch.cols*2,0,patch.cols,patch.rows)));
    }

    if (show){imshow("Patch ", out); waitKey();}
    return out;
}


Mat showTriplet(const Mat &p0,const Mat &p1,const Mat &p2,const Mat &p3,const Mat &p4, bool show /*= true*/)
{
    vector<Mat> ps;

   ps.push_back(showRGBDPatch(p0,false));
   ps.push_back(showRGBDPatch(p1,false));
   ps.push_back(showRGBDPatch(p2,false));
   ps.push_back(showRGBDPatch(p3,false));
   ps.push_back(showRGBDPatch(p4,false));

   Mat out(ps.back().rows*5,ps.back().cols,CV_32FC3);
   for (int i=0; i < 5; ++i)
       ps[i].copyTo(out(Rect(0,ps.back().rows*i,ps.back().cols,ps.back().rows)));

    if (show){imshow("Patch ", out); waitKey();}
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
        b[0] += fi*delta;
        b[1] += fj*delta;
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
                normal(2) = (d+ddx*(c-ox+1))*(d+ddy*(r-oy+1) -ddx*ddy*(ox-c)*(oy-r));
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

Mat growForeground(Mat &depth)
{
    auto check = [](float ref, float other)->bool
    {
        return abs(ref-other)<0.01f;
    };

    Mat mask = Mat::zeros(depth.size(),CV_8U);

    stack<Point> cands;
    cands.push(Point(depth.cols/2,depth.rows/2));
//    cands.push(Point(0,0));

    while (!cands.empty())
    {
        Point p = cands.top();
        cands.pop();
        mask.at<uchar>(p) = 255;   // Mark visited

        float d = depth.at<float>(p);
        if ((p.x>0) && (!mask.at<uchar>(p.y,p.x-1)))
            if (check(d,depth.at<float>(p.y,p.x-1))) cands.push(Point(p.x-1,p.y));
        if ((p.y>0) && (!mask.at<uchar>(p.y-1,p.x)))
            if (check(d,depth.at<float>(p.y-1,p.x))) cands.push(Point(p.x,p.y-1));

        if ((p.x<depth.cols-1) && (!mask.at<uchar>(p.y,p.x+1)))
            if (check(d,depth.at<float>(p.y,p.x+1))) cands.push(Point(p.x+1,p.y));
        if ((p.y<depth.rows-1) && (!mask.at<uchar>(p.y+1,p.x)))
            if (check(d,depth.at<float>(p.y+1,p.x))) cands.push(Point(p.x,p.y+1));

    }

    imshow("depth",depth);imshow("mask",mask); waitKey();
    return mask;

}

}
