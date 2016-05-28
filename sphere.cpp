
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "painter.h"
#include "sphere.h"


using namespace std;


SphereRenderer::SphereRenderer(Matrix3f &cam){init(cam);}
SphereRenderer::SphereRenderer(){m_camera.setZero();}

void SphereRenderer::init(Matrix3f &cam)
{
    m_camera = cam;
}
/***************************************************************************/
Isometry3f SphereRenderer::createTransformation(Vector3f &vertex,float scale,float angle)
{
    Matrix3f rot = computeRotation(vertex);
    Isometry3f transform = Isometry3f::Identity();
    transform.linear() = AngleAxisf(angle*CV_PI/180.0,Vector3f(0,0,1))*rot;
    transform.translation() = Vector3f(0,0,scale);
    return transform;
}

/***************************************************************************/
pair<int,int> SphereRenderer::renderView(Model &model, Isometry3f &pose, Mat &col, Mat &dep, bool clipped)
{
    Painter painter;

    // Assume to render a full image
    int x=0,y=0,w=0,h=0;

    // If we only want to render a small a view
    if (clipped)
    {
        // Figure out how the projected 3D bounding box of the rendered object would lie on the image plane
        Matrix<float,3,8> tbb = m_camera*(pose*model.boundingBox);
        tbb.row(0).array() /= tbb.row(2).array();
        tbb.row(1).array() /= tbb.row(2).array();
        x = max(0.0f,tbb.row(0).minCoeff());
        y = max(0.0f,tbb.row(1).minCoeff());
        w = max(0,min(painter.getWidth(), (int)tbb.row(0).maxCoeff())-x);
        h = max(0,min(painter.getHeight(),(int)tbb.row(1).maxCoeff())-y);
    }

    RealWorldCamera cam(m_camera,pose);
    painter.clearObjects();
    painter.setBackground(0,0,0);
    painter.addPaintObject(&cam);
    painter.addPaintObject(&model);
    painter.paint(x,y,w,h);
    painter.copyColorTo(col);
    painter.copyDepthTo(dep);
    //imshow("rendering_col",col);imshow("rendering_dep",dep);waitKey();
    return {x,y};
}
/***************************************************************************/
vector<RenderView, Eigen::aligned_allocator<RenderView> > SphereRenderer::createViews(Model &model,int sphereDep,Vector3f scale, Vector3f rotation, bool skipLowerHemi, bool skipRearPart, bool clip)
{
    assert(m_camera(0,0)!=0);

    vector<float> sca,rots;
    for (float i = rotation(0); i <= rotation(2); i += rotation(1)) rots.push_back(i);
    for (float i = scale(0);i <= scale(2); i += scale(1)) sca.push_back(i);

    vector<RenderView, Eigen::aligned_allocator<RenderView> > out;

    vector<Vector3f> sphere = initSphere(sphereDep);
    for(float currsca : sca)
        for(Vector3f &pos : sphere)
        {
            if((pos(0) < 0) && skipRearPart) continue;  // Skip negative x-part (for symmetric objects)
            if((pos(2) < 0) && skipLowerHemi) continue;  // Skip the lower hemisphere of the object
            for(float curr_rot : rots)
            {
                RenderView view;
                view.pose = createTransformation(pos,currsca,curr_rot);
                pair<int,int> render = renderView(model,view.pose,view.col,view.dep, clip);
                view.x_off = render.first;
                view.y_off = render.second;
                out.push_back(view);
            }
        }

    /*
    for(int i=0; i< sphere.size()-1; ++i)
    {
        Matrix3f roti = createTransformation(sphere[i],1,0).linear(),rotj = createTransformation(sphere[i+1],1,0).linear();
        cerr << 180/M_PI*acosf(sphere[i].dot(sphere[i+1])) << " " << Quaternionf(roti).angularDistance(Quaternionf(rotj)) << endl;
    }
    */

    clog << "    - sphereRenderer - rendered " << out.size() << " views" << endl;
    return out;
}

/***************************************************************************/
Matrix3f SphereRenderer::computeRotation(Vector3f &eye)
{
    Vector3f up(0,0,1);
    if(eye(0)==0&&eye(1)==0&&eye(2)!=0) up = Vector3f(-1,0,0);
    Matrix3f rot;
    rot.col(2) = -eye.normalized();
    rot.col(0) = (rot.col(2).cross(up.normalized())).normalized();
    rot.col(1) = rot.col(0).cross(-rot.col(2));
    return rot.transpose();
}


/***************************************************************************/
Matrix3f SphereRenderer::getRot(Vector3f center,Vector3f eye,Vector3f up)
{
    Matrix3f rot;
    rot.col(2) = (center-eye).normalized();
    rot.col(0) = (rot.col(2).cross(up.normalized())).normalized();
    rot.col(1) = rot.col(0).cross(-rot.col(2));
    return rot.transpose();
}
/***************************************************************************/

void SphereRenderer::subdivide(vector<Vector3f> &sphere,Vector3f v1,Vector3f v2,Vector3f v3,int depth)
{
    if(depth==0)
    {
        int flag1=-1,flag2=-1,flag3=-1;
        for(uint i=0;i<sphere.size();++i)
        {
            if(sphere[i]==v1)flag1=i;
            if(sphere[i]==v2)flag2=i;
            if(sphere[i]==v3)flag3=i;
            if(flag1!=-1&&flag2!=-1&&flag3!=-1) break;
        }
        if(flag1==-1) sphere.push_back(v1);
        if(flag2==-1) sphere.push_back(v2);
        if(flag3==-1) sphere.push_back(v3);
        return;
    }
    Vector3f v12 = (v1+v2).normalized();
    Vector3f v23 = (v2+v3).normalized();
    Vector3f v31 = (v3+v1).normalized();
    subdivide(sphere,v1,v12,v31,depth-1);
    subdivide(sphere,v2,v23,v12,depth-1);
    subdivide(sphere,v3,v31,v23,depth-1);
    subdivide(sphere,v12,v23,v31,depth-1);
}

/***************************************************************************/

vector<Vector3f> SphereRenderer::initSphere(int depth)
{
    vector<Vector3f> sphere;

    float X = 0.525731112119133606f, Z = 0.850650808352039932f;
    vector< vector<int> > ind = {
        {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
        {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
        {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
        {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };

    vector<Vector3f> v(12);
    v[0]=Vector3f(-X,0.0,+Z).normalized();
    v[1]=Vector3f(+X,0.0,+Z).normalized();
    v[2]=Vector3f(-X,0.0,-Z).normalized();
    v[3]=Vector3f(+X,0.0,-Z).normalized();
    v[4]=Vector3f(0.0,+Z,+X).normalized();
    v[5]=Vector3f(0.0,+Z,-X).normalized();
    v[6]=Vector3f(0.0,-Z,+X).normalized();
    v[7]=Vector3f(0.0,-Z,-X).normalized();
    v[8]=Vector3f(+Z,+X,0.0).normalized();
    v[9]=Vector3f(-Z,+X,0.0).normalized();
    v[10]=Vector3f(+Z,-X,0.0).normalized();
    v[11]=Vector3f(-Z,-X,0.0).normalized();

    for(uint i=0;i<20;i++)
        subdivide(sphere,v[ind[i][0]],v[ind[i][1]],v[ind[i][2]],depth);

    float min=361;
    for(uint i=0;i<sphere.size();++i)
        for(uint j=0;j<sphere.size();++j)
            if(i!=j)
            {
                float val = 180/M_PI*std::acos(sphere[i].dot(sphere[j]));
                if(min>val) min=val;
            }

    for (Vector3f &p : sphere) p.normalize();

    clog << "    - sphereRenderer - vertices: " << sphere.size() << "   min: " << min << " deg apart" << endl;
    return sphere;
}

/***************************************************************************/

vector<Vector3f> SphereRenderer::initSphere(int inc_steps, int azi_steps)
{
    vector<Vector3f> sphere;

    // Push first vertex already (because up-vector not defined) and do the rest
    sphere.push_back(Vector3f(0,0,1));
    float inc=M_PI/inc_steps, azi=0;
    for(int inc_=1; inc_ < inc_steps; inc += M_PI/inc_steps, inc_++)
        for(int azi_=0; azi_ < azi_steps; azi += 2*M_PI/azi_steps, azi_++)
        {
            sphere.push_back(Vector3f(sin(inc)*cos(azi),sin(inc)*sin(azi),cos(inc)));
        }

    return sphere;
}

/************************ END OF FILE **************************************/
