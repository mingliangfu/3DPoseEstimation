
#include <chrono>
#include <set>
#include <iostream>
#include <fstream>

#include <Eigen/Geometry>
#include <Eigen/Cholesky>

#include <boost/filesystem.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>

//add
//#include <opencv2/viz/widget_accessor.hpp>


#include "../include/icp.h"
#include "../include/utilities.h"


using namespace std;
using namespace boost;

#define BORDER 3

#define SQR(x) ((x)*(x))


ICP::ICP(){}


void ICP::dump3DTransform(string file)
{
    ofstream out(file, ios::binary);
    out.write((char*) &m_step, sizeof(float));
    out.write((char*) dims.data(), 3*sizeof(float));
    for(IcpVoxel &v : voxel_grid)
    {
        out.write((char*) v.closestPoint.data(), 3*sizeof(float));
        out.write((char*) v.closestNormal.data(), 3*sizeof(float));
    }

}

bool ICP::load3DTransform(string file)
{
    if (!filesystem::exists(filesystem::path(file)))
        return false;

    ifstream in(file, ios::binary);
    in.read((char*) &m_step, sizeof(float));
    in.read((char*) dims.data(), 3*sizeof(float));
    voxel_grid.resize(dims(0)*dims(1)*dims(2));
    for(IcpVoxel &v : voxel_grid)
    {
        in.read((char*) v.closestPoint.data(), 3*sizeof(float));
        in.read((char*) v.closestNormal.data(), 3*sizeof(float));
    }
    return true;

}

void ICP::setData(Model &model)
{

    points = model.getPoints();

    Vector3f min = model.bb_min, max = model.bb_max;
    m_radius = (max-min).norm()*0.5;

    // Move the object a bit further into the grid by expanding min and max
    m_min=min-(max-min)/3;
    m_max=max+(max-min)/3;

    // Check if distance transform has already been loaded
    if(!voxel_grid.empty())
    {
        cerr << "ICP - num " << dims.transpose() << " with step size " << m_step << "  (DUMP)" <<endl;
        return;
    }

    // Figure out dimensions
    Vector3f diff = m_max-m_min;
    float num_x=pow(pow(100.0,3.0)*(diff(0)*diff(0))/(diff(1)*diff(2)),0.33f);
    dims(0)=static_cast<int>(num_x+0.5f);
    dims(1)=static_cast<int>(num_x*diff(1)/diff(0) + 0.5f);
    dims(2)=static_cast<int>(num_x*diff(2)/diff(0) + 0.5f);
    m_step = diff.cwiseQuotient(dims.cast<float>()).maxCoeff();
    dims =  (diff/m_step).cast<int>();
    cerr << "ICP - num " << dims.transpose() << " with step size " << m_step << endl;

    // Initialize
    voxel_grid.resize(dims(0)*dims(1)*dims(2));
    for(IcpVoxel &v : voxel_grid)
    {
        v.ds = dims;
        v.distance = v.ds.squaredNorm();
        v.closestPoint = Vector3f(0,0,0);
        v.closestNormal = Vector3f(0,0,0);
    }

    int skippedFaces=0;
    if(!model.getFaces().empty())
    {
        for(Vec3i &f : model.getFaces())
        {

            Vector3f normal = (model.getPoints()[f(1)]-model.getPoints()[f(0)]).cross(model.getPoints()[f(2)]-model.getPoints()[f(0)]);
            normal.normalize();
            if(!normal.allFinite()) // Sometimes the normal can be NaN by construction... skip it!
            {
                skippedFaces++;
                continue;
            }

            // Compute point as being the triangle center
            Vector3f point = (model.getPoints()[f(0)]+model.getPoints()[f(1)]+model.getPoints()[f(2)])/3.0f;
            Vector3i voxel = ((point-m_min)/m_step).cast<int>();
            int index = computeIndex(voxel);
            voxel_grid[index].distance=0;
            voxel_grid[index].ds=voxel;
            voxel_grid[index].closestPoint=point;
            voxel_grid[index].closestNormal=normal;
        }
    }
    else
    {
        cerr << "ICP - load point cloud" << endl;
        for(uint i=0;i<model.getPoints().size();++i)
        {
            Vector3f point = model.getPoints()[i];
            Vector3i voxel = ((point-m_min)/m_step).cast<int>();
            int index = computeIndex(voxel);
            voxel_grid[index].distance=0;
            voxel_grid[index].ds=voxel;
            voxel_grid[index].closestPoint=point;
            voxel_grid[index].closestNormal=model.getNormals()[i];
        }
    }

    if(skippedFaces>0) cerr << "ICP - skipped " << skippedFaces << " invalid faces!" << endl;

    // http://www.robots.ox.ac.uk/~vgg/publications/papers/fitzgibbon01c.pdf
    compute3DDistanceTransform();
    for(IcpVoxel &i : voxel_grid) i.distance = m_step*sqrt(i.distance);

}


/***************************************************************************/

Isometry3f ICP::computePointPlane(vector<Vector3f> &src,vector<Vector3f> &dst,vector<Vector3f> &nor)
{
    assert(src.size()==dst.size() && src.size()==nor.size());

    // Maybe also have a look at that?
    // https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf

    // http://www.cs.princeton.edu/~smr/papers/icpstability.pdf
    Matrix<float,6,6> C;
    Matrix<float,6,1> b;
    C.setZero();
    b.setZero();
    for(uint i=0;i<src.size();++i)
    {
        Vector3f cro = src[i].cross(nor[i]);
        C.block<3,3>(0,0) += cro*cro.transpose();
        C.block<3,3>(0,3) += nor[i]*cro.transpose();
        C.block<3,3>(3,3) += nor[i]*nor[i].transpose();

        float sum = (src[i]-dst[i]).dot(nor[i]);
        b.head(3) -= cro*sum;
        b.tail(3) -= nor[i]*sum;
    }
    C.block<3,3>(3,0) = C.block<3,3>(0,3);
    Matrix<float,6,1> x = C.ldlt().solve(b);

    Isometry3f transform = Isometry3f::Identity();
    transform.linear() =
            (AngleAxisf(x(0), Vector3f::UnitX())
             * AngleAxisf(x(1), Vector3f::UnitY())
             * AngleAxisf(x(2), Vector3f::UnitZ())).toRotationMatrix();
    transform.translation() = x.block(3,0,3,1);

    return transform;
}

/***************************************************************************/

Isometry3f ICP::computePointPoint(vector<Vector3f> &src,vector<Vector3f> &dst,Vector3f &a,Vector3f &b,bool withScale)
{
    assert(src.size()==dst.size());

    // http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Zinsser05-PSR.pdf

    Matrix3f K = Matrix3f::Zero();
    for (uint i=0; i < src.size();i++)
        K += (b-dst[i])*(a-src[i]).transpose();

    JacobiSVD<Matrix3f> svd(K, ComputeFullU | ComputeFullV);
    Matrix3f R = svd.matrixU()*svd.matrixV().transpose();
    if(R.determinant()<0) R.col(2) *= -1;

    if (withScale)
    {
        float s_up=0,s_down=0;
        for (uint i=0; i < src.size();i++)
        {
            Vector3f b_tilde = (b-dst[i]);
            Vector3f a_tilde = R*(a-src[i]);
            s_up += b_tilde.dot(a_tilde);
            s_down += a_tilde.dot(a_tilde);
        }
        R *= s_up/s_down;
    }

    Isometry3f transform = Isometry3f::Identity();
    transform.linear() = R;
    transform.translation() = b - R*a;
    return transform;
}
/***************************************************************************/

Isometry3f ICP::computePointPoint(vector<Vector3f> &src,vector<Vector3f> &dst)
{
    assert(src.size()==dst.size());
    Vector3f a(0,0,0), b(0,0,0);
    for(Vector3f &p : src) a += p;
    for(Vector3f &p : dst) b += p;
    a /= src.size();
    b /= src.size();
    return computePointPoint(src,dst,a,b,false);
}

/***************************************************************************/
IcpVoxel& ICP::getNN(Vector3f &p)
{
    float step=1.0f/m_step;
    int x = (p(0)-m_min(0))*step, y = (p(1)-m_min(1))*step, z = (p(2)-m_min(2))*step;
    if(x<BORDER||x>=dims(0)-BORDER-1||y<BORDER||y>=dims(1)-BORDER-1||z<BORDER||z>=dims(2)-BORDER-1) return voxel_grid[0];
    return voxel_grid[computeIndex(x,y,z)];
}

/***************************************************************************/

IcpStruct ICP::compute(vector<Vector3f, Eigen::aligned_allocator<Vector3f> > &in,
                       int iterations,float error_delta, bool point2plane)
{

    IcpStruct icp;

    float distThreshold=m_radius;
    float step=1.0f/m_step;
    float prev_error=numeric_limits<float>::max();
    int prev_inlier=0;

    vector<Vector3f> src,dst,nor;
    src.reserve(in.size());
    dst.reserve(in.size());
    nor.reserve(in.size());

    //#define ICP_DEBUG

#ifdef ICP_DEBUG
    viz::Viz3d window("ICP");
    cout << "aaaaaaaaaabbbbbbbbbb "<< endl ;
#endif

    for(int i=0;i<iterations;i++)
    {
        icp.error  = 0;
        icp.inlier = 0;
        icp.iter = i;
        src.clear();
        dst.clear();
        nor.clear();

        // Distance-transform NN
        for(Vector3f &p : in)
        {

            // Compute voxel position and skip if close to border or outside of grid
            int x = (p(0)-m_min(0))*step, y = (p(1)-m_min(1))*step, z = (p(2)-m_min(2))*step;
            if(x<BORDER||x>=dims(0)-BORDER-1||y<BORDER||y>=dims(1)-BORDER-1||z<BORDER||z>=dims(2)-BORDER-1) continue;

            int index=computeIndex(x,y,z);
            Vector3f &dst_pt=voxel_grid[index].closestPoint;

            float dist = (p-dst_pt).norm();
            if(dist > distThreshold) continue;
            icp.error+=dist;

            src.push_back(p);
            dst.push_back(dst_pt);
            nor.push_back(voxel_grid[index].closestNormal);
        }

        if(src.size()==0) return IcpStruct();


        icp.inlier=src.size();
        icp.error/=src.size();
        distThreshold = icp.error*3;


#ifdef ICP_DEBUG
        window.removeAllWidgets();
        Mat src_mat(src.size(),1,CV_32FC3, src.data());
        Mat dst_mat(dst.size(),1,CV_32FC3, dst.data());
        Mat nor_mat(nor.size(),1,CV_32FC3, nor.data());
        Mat obj_mat(points.size(),1,CV_32FC3, points.data());
        window.setBackgroundColor(viz::Color::white());
        window.showWidget("src",viz::WCloud(src_mat,viz::Color::red()));
        window.showWidget("dst",viz::WCloud(dst_mat,viz::Color::blue()));
        window.showWidget("obj",viz::WCloud(obj_mat,viz::Color::black()));
        window.showWidget("normals",viz::WCloudNormals(dst_mat,nor_mat,1,0.01,viz::Color::black()));
        window.setRenderingProperty("src",viz::POINT_SIZE,2);
        window.setRenderingProperty("dst",viz::POINT_SIZE,2);
        for (uint i=0; i < src.size(); i++)
        {
            stringstream ss; ss << i;
            window.showWidget(ss.str(),viz::WLine(src_mat.at<Point3f>(i,1),dst_mat.at<Point3f>(i,1),viz::Color::green()));
        }
        cout << "aaaaaaaaaacccccccccc "<< endl ;
        //window.spinOnce(1000);
        window.spin();
#endif


        if(std::abs(prev_error-icp.error)<error_delta && prev_inlier == icp.inlier) break;

        prev_error = icp.error;
        prev_inlier= icp.inlier;
        
        Isometry3f update = point2plane ? computePointPlane(src,dst,nor) : computePointPoint(src,dst);

        for(Vector3f &p : in) p = update*p;
        icp.pose = update*icp.pose;

    }

    return icp;
}

/***************************************************************************/

/***************************************************************************/

IcpStruct ICP::computeWithOutliers(vector<Vector3f> &in, int iterations,float error_delta, bool point2plane)
{

    IcpStruct icp;

    float step=1.0f/m_step;
    float prev_error=numeric_limits<float>::max();
    int prev_inlier=0;

    vector<Vector3f> src,dst,nor;
    src.reserve(in.size());
    dst.reserve(in.size());
    nor.reserve(in.size());

    for(int i=0;i<iterations;i++)
    {
        icp.error  = 0;
        icp.inlier = 0;
        icp.iter = i;
        src.clear();
        dst.clear();
        nor.clear();

        // Distance-transform NN
        for(Vector3f &p : in)
        {
            // Compute voxel position and skip if close to border or outside of grid
            int x = (p(0)-m_min(0))*step, y = (p(1)-m_min(1))*step, z = (p(2)-m_min(2))*step;
            if(x<BORDER||x>=dims(0)-BORDER-1||y<BORDER||y>=dims(1)-BORDER-1||z<BORDER||z>=dims(2)-BORDER-1) continue;

            int index=computeIndex(x,y,z);
            Vector3f &dst_pt=voxel_grid[index].closestPoint;
            icp.error+=(p-dst_pt).norm();
            src.push_back(p);
            dst.push_back(dst_pt);
            nor.push_back(voxel_grid[index].closestNormal);
        }

        if(src.size()==0) return IcpStruct();

        icp.inlier=src.size();
        icp.error/=src.size();

        if(fabs(prev_error-icp.error)<error_delta && prev_inlier == icp.inlier) break;

        prev_error = icp.error;
        prev_inlier= icp.inlier;

        Isometry3f update = point2plane ? computePointPlane(src,dst,nor) : computePointPoint(src,dst);

        for(Vector3f &p : in) p = update*p;
        icp.pose = update*icp.pose;
    }

    return icp;
}
/***************************************************************************/

void ICP::compute3DDistanceTransform()
{
    int n=BORDER;
    bool change=true;
    int pass=0;
    while(change)
    {
        change=false;

        //Forward
        for(int z=n;z<dims(2)-n;++z)
            for(int y=n;y<dims(1)-n;++y)
                for(int x=n;x<dims(0)-n;++x)
                {
                    int index = computeIndex(x,y,z);
                    for(int k=-n;k<=n;++k)
                        for(int j=-n;j<=n;++j)
                            for(int i=-n;i<=n;++i)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d<voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance  = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }

        //Backward
        for(int z=dims(2)-1-n;z>=n;--z)
            for(int y=dims(1)-1-n;y>=n;--y)
                for(int x=dims(0)-1-n;x>=n;--x)
                {
                    int index = computeIndex(x,y,z);
                    for(int k=-n;k<=n;++k)
                        for(int j=-n;j<=n;++j)
                            for (int i=-n;i<=n;++i)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d < voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }
        pass++;
        if (!change) break;

        //Forward
        for(int z=n;z<dims(2)-n;++z)
            for(int x=n;x<dims(0)-n;++x)
                for(int y=n;y<dims(1)-n;++y)
                {
                    int index = computeIndex(x,y,z);
                    for(int k=-n;k<=n;++k)
                        for(int i=-n;i<=n;++i)
                            for(int j=-n;j<=n;++j)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d<voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance  = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }

        //Backward
        for(int z=dims(2)-1-n;z>=n;--z)
            for(int x=dims(0)-1-n;x>=n;--x)
                for(int y=dims(1)-1-n;y>=n;--y)
                {
                    int index = computeIndex(x,y,z);
                    for(int k=-n;k<=n;++k)
                        for (int i=-n;i<=n;++i)
                            for(int j=-n;j<=n;++j)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d < voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }
        pass++;
        if (!change) break;

        //Forward
        for(int x=n;x<dims(0)-n;++x)
            for(int y=n;y<dims(1)-n;++y)
                for(int z=n;z<dims(2)-n;++z)
                {
                    int index = computeIndex(x,y,z);
                    for(int i=-n;i<=n;++i)
                        for(int j=-n;j<=n;++j)
                            for(int k=-n;k<=n;++k)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d<voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance  = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }

        //Backward
        for(int x=dims(0)-1-n;x>=n;--x)
            for(int y=dims(1)-1-n;y>=n;--y)
                for(int z=dims(2)-1-n;z>=n;--z)
                {
                    int index = computeIndex(x,y,z);
                    for (int i=-n;i<=n;++i)
                        for(int j=-n;j<=n;++j)
                            for(int k=-n;k<=n;++k)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d < voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }
        pass++;
        if (!change) break;

        //Forward
        for(int x=n;x<dims(0)-n;++x)
            for(int z=n;z<dims(2)-n;++z)
                for(int y=n;y<dims(1)-n;++y)
                {
                    int index = computeIndex(x,y,z);
                    for(int i=-n;i<=n;++i)
                        for(int k=-n;k<=n;++k)
                            for(int j=-n;j<=n;++j)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d<voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance  = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }

        //Backward
        for(int x=dims(0)-1-n;x>=n;--x)
            for(int z=dims(2)-1-n;z>=n;--z)
                for(int y=dims(1)-1-n;y>=n;--y)
                {
                    int index = computeIndex(x,y,z);
                    for (int i=-n;i<=n;++i)
                        for(int k=-n;k<=n;++k)
                            for(int j=-n;j<=n;++j)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d < voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }
        pass++;
        if (!change) break;

        //Forward
        for(int y=n;y<dims(1)-n;++y)
            for(int x=n;x<dims(0)-n;++x)
                for(int z=n;z<dims(2)-n;++z)
                {
                    int index = computeIndex(x,y,z);
                    for(int j=-n;j<=n;++j)
                        for(int i=-n;i<=n;++i)
                            for(int k=-n;k<=n;++k)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d<voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance  = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }

        //Backward
        for(int y=dims(1)-1-n;y>=n;--y)
            for(int x=dims(0)-1-n;x>=n;--x)
                for(int z=dims(2)-1-n;z>=n;--z)
                {
                    int index = computeIndex(x,y,z);
                    for(int j=-n;j<=n;++j)
                        for (int i=-n;i<=n;++i)
                            for(int k=-n;k<=n;++k)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d < voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }
        pass++;
        if (!change) break;

        //Forward
        for(int y=n;y<dims(1)-n;++y)
            for(int z=n;z<dims(2)-n;++z)
                for(int x=n;x<dims(0)-n;++x)
                {
                    int index = computeIndex(x,y,z);
                    for(int j=-n;j<=n;++j)
                        for(int k=-n;k<=n;++k)
                            for(int i=-n;i<=n;++i)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d<voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance  = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }

        //Backward
        for(int y=dims(1)-1-n;y>=n;--y)
            for(int z=dims(2)-1-n;z>=n;--z)
                for(int x=dims(0)-1-n;x>=n;--x)
                {
                    int index = computeIndex(x,y,z);
                    for(int j=-n;j<=n;++j)
                        for(int k=-n;k<=n;++k)
                            for (int i=-n;i<=n;++i)
                            {
                                int index1 = computeIndex(x+i,y+j,z+k);
                                float d = (voxel_grid[index1].ds-Vector3i(x,y,z)).squaredNorm();
                                if(d < voxel_grid[index].distance)
                                {
                                    voxel_grid[index].distance = d;
                                    voxel_grid[index].ds = voxel_grid[index1].ds;
                                    voxel_grid[index].closestPoint  = voxel_grid[index1].closestPoint;
                                    voxel_grid[index].closestNormal = voxel_grid[index1].closestNormal;
                                    change=true;
                                }
                            }
                }
        pass++;
        if (!change) break;

    }
    cerr << "ICP - has changed in pass " << pass << endl;


#if 0
    vector<Vector3f> points, normals;
    for(int x=0;x<dims(0);++x)
        for(int y=0;y<dims(1);++y)
            for(int z=0;z<dims(2);++z)
            {
                int index = computeIndex(x,y,z);
                if(voxel_grid[index].distance>0) continue;
                points.push_back(voxel_grid[index].closestPoint);
                normals.push_back(voxel_grid[index].closestNormal);
            }
    cv::viz::Viz3d show;
    cv::Mat pointsMat(1,points.size(),CV_32FC3, points.data());
    cv::Mat normalsMat(1,normals.size(),CV_32FC3, normals.data());
    show.showWidget("cloud",viz::WCloud(pointsMat,cv::viz::Color::green(),normalsMat));
    show.showWidget("normals",cv::viz::WCloudNormals(pointsMat,normalsMat,12,0.01));
    show.spin();
#endif



    return;
}

