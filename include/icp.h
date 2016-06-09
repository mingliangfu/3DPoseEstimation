
#pragma once


#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "model.h"

struct IcpVoxel
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector3f closestPoint,closestNormal;
    Vector3i ds;
    float distance;
};

struct IcpStruct
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IcpStruct()
    {
        error = numeric_limits<float>::max();
        inlier=iter=0;
        pose = Isometry3f::Identity();
    }

    Isometry3f pose;
    float error;
    int inlier,iter;
};


class ICP
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    ICP();

    void setData(Model &model);

    void dump3DTransform(string file);
    bool load3DTransform(string file);

    IcpStruct compute(vector<Vector3f, Eigen::aligned_allocator<Vector3f> > &points,
                      int iterations,float error_delta, bool point2plane=true);
    IcpStruct computeWithOutliers(vector<Vector3f> &points,int iterations,float error_delta, bool point2plane=true);

    Isometry3f computePointPlane(vector<Vector3f> &src,vector<Vector3f> &dst,vector<Vector3f> &nor);
    Isometry3f computePointPoint(vector<Vector3f> &src,vector<Vector3f> &dst,Vector3f &cen_src,Vector3f &cen_dst, bool withScale=false);
    Isometry3f computePointPoint(vector<Vector3f> &src,vector<Vector3f> &dst);

    IcpVoxel& getNN(Vector3f &p);


private:

    void compute3DDistanceTransform();

    inline int computeIndex(Vector3i &pos){return computeIndex(pos(0),pos(1),pos(2));}
    inline int computeIndex(int x, int y, int z) {int val = x + dims(0)*y + dims(0)*dims(1)*z;
                                                  //assert(val>=0 && val < m_num(0)*m_num(1)*m_num(2));
                                                  return val;}

    Mat points_mat;
    vector<Vector3f, Eigen::aligned_allocator<Vector3f> > points;

    // Number of voxels for all axes
    Vector3i dims;
    //The value [m] that corresponds to the smallest/largest voxel on the axes.
    Vector3f m_min,m_max;

    //The radius of the bounding box of the model and voxel size.
    float m_radius, m_step;

    //The voxel grid that stores for each voxel the closest occupied voxel.
    //This grid is used to compute the nearest neighbor efficiently.
    vector<IcpVoxel, Eigen::aligned_allocator<IcpVoxel> > voxel_grid;


};


