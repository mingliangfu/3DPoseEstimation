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

}
