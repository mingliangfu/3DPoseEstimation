
#pragma once

#include <QtOpenGL>
#include <QOpenGLFunctions>


#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <opencv2/core.hpp>

//#include <GL/glew.h>


#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif



using namespace cv;
using namespace Eigen;
using namespace std;

namespace Gopnik {



class PaintObject
{
public:
    virtual void paint(void)=0;
};


class SingletonPainter : public QGLWidget, protected QOpenGLFunctions
{

public:
	
    SingletonPainter(float near,float far,int width,int height);
    ~SingletonPainter();

    int getHeight(){return m_height;}
    int getWidth(){return m_width;}
    float getNear(){return m_near;}
    float getFar(){return m_far;}

    void clearBackground(float r,float g,float b);
    void clearObjects(){m_objects.clear();}

    void addPaintObject(PaintObject *object){m_objects.push_back(object);}
	
    void paint(int x=0,int y=0,int w=0,int h=0);
    void paint(Rect &rect);

    void bindVBOs(vector<Vector3f> &vertex,vector<Vector3i> &faces, GLuint &vert, GLuint &ind);
    void drawVBOs(GLuint vert, GLuint ind, int count);

    inline void copyColorTo(Mat &dest){m_color(copy_rect).copyTo(dest);}
    inline void copyDepthTo(Mat &dest){m_depth(copy_rect).copyTo(dest);}

protected:
    
    void convertZBufferToDepth(Mat &depth);
    void resizeGL(int w,int h);
	void paintGL();

private:
    
    // Rect in the image which should be rendered.
    Rect render_rect, copy_rect;
    //OpenGL near/far.
    float m_near,m_far;
    // OpenGL viewport dimensions
    int m_height, m_width;

    //Background RGB color
    Vector3f m_background;
    //The rendered depth and color buffer is stored here.
    Mat m_depth,m_color;
    //Vector of objects that is painted (in this order).
    vector<PaintObject*> m_objects;

    //QT framebuffer object for offline rendering.
    QOpenGLFramebufferObject *m_fbo;

    unsigned int fbo; // The frame buffer object
    unsigned int fbo_depth; // The depth buffer for the frame buffer object
    unsigned int fbo_color; // The texture object to write our frame buffer object to
};


class Painter// : protected QOpenGLFunctions
{

public:
	
    Painter(){}
    ~Painter(){}

    int getHeight(){return getSingleton()->getHeight();}
    int getWidth(){return getSingleton()->getWidth();}
    float getNear(){return getSingleton()->getNear();}
    float getFar(){return getSingleton()->getFar();}
    inline float getAspect(){return static_cast<float>(getWidth())/static_cast<float>(getHeight());}

    void setBackground(Vector3f &col){getSingleton()->clearBackground(col(0),col(1),col(2));}
    void setBackground(float r,float g,float b){getSingleton()->clearBackground(r,g,b);}
    void clearObjects(){getSingleton()->clearObjects();}

    inline void addPaintObject(PaintObject *object){getSingleton()->addPaintObject(object);}
	
    inline void paint(int x=0,int y=0,int w=0,int h=0){getSingleton()->paint(x,y,w,h);}
    inline void paint(Rect &rect){getSingleton()->paint(rect);}

    inline void copyColorTo(Mat &dest){getSingleton()->copyColorTo(dest);}
    inline void copyDepthTo(Mat &dest){getSingleton()->copyDepthTo(dest);}

    static SingletonPainter* getSingleton();

private:
	
    //We need only ony QT framebuffer object in which we paint
    static SingletonPainter *m_singleton;
};


class RealWorldCamera : public PaintObject
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    virtual ~RealWorldCamera(){}
    RealWorldCamera(Matrix3f &kma,Isometry3f &transform);
    void paint();
    
    // OpenGL near/far
    float m_near,m_far;
    // Rotation and translation matrix.
    Isometry3f m_pose;
    // Internal camera matrix.
    Matrix3f m_cam;
};


class BackgroundCamera : public PaintObject
{
public:

    BackgroundCamera(Mat &background);

    void paint();

    Mat m_background; //This image stores the background image.
};

class CoordinateSystem : public PaintObject
{

public:

    CoordinateSystem(float size);

    void paint();

    float m_size;  // Size of the coordinate axis
    vector<Vector3f> m_points;
    vector<GLushort> m_indices;

};

class AxisAlignedPlane : public PaintObject
{
public:

    AxisAlignedPlane(int axis, float offset, float r,float g, float b);

    void paint();

    float m_r, m_g, m_b;    // Color of plane
    vector<Vector3f> m_points;
    vector<GLushort> m_indices;

};


class BoundingBox : public PaintObject
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    BoundingBox(Matrix<float,3,8> &in_bb);
    virtual ~BoundingBox(){}

    void paint();

    Matrix<float,3,8> m_bb;  // Bounding box
    vector<GLushort> m_indices;

};


}
