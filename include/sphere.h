#pragma once

#include <list>
#include <vector>
#include <iomanip>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

#include "model.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace sz { // For Wadim

struct RenderView
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Mat col, dep;
    int x_off, y_off;
    Isometry3f pose;
};


class SphereRenderer
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3f m_camera;


    SphereRenderer(Matrix3f &cam);
    SphereRenderer();
    void init(Matrix3f &cam);

    pair<int,int> renderView(Model &model, Isometry3f &pose, Mat &col, Mat &dep,bool clipped=true);

    vector<RenderView, Eigen::aligned_allocator<RenderView> > createViews(Model &object, int sphereDep, Vector3f scale, Vector3f rot, bool skipLowerHemi=true, bool clip=true);

    Isometry3f createTransformation(Vector3f &sphere_pt,float scale,float rotation);

    void subdivide(vector<Vector3f> &sphere,Vector3f v1,Vector3f v2,Vector3f v3,int depth);

    vector<Vector3f> initSphere(int depth);
    vector<Vector3f> initSphere(int inc_steps, int azi_steps);

    Matrix3f computeRotation(Vector3f &vertex);

    Matrix3f getRot(Vector3f center,Vector3f eye,Vector3f up);   

};

}


