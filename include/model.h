#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "painter.h"

using namespace cv;
using namespace std;
using namespace Eigen;


namespace sz { // For Wadim

class Model : public PaintObject
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Model(float cube_size);
    Model();
    ~Model();

    void paint();
    void bindVBOs();

    void computeBoundingBox();

    float computeMeshResolution();

    void computeVertexNormals();

    void subsampleCloud(float leaf_size);

    void computeLocalCoordsColors();

    vector<bool> computeEdgePoints();

    vector<Vector3f> &getPoints() {return m_points;}
    vector<Vector3f> &getColors() {return m_colors;}
    vector<Vector3f> &getNormals(){return m_normals;}
    vector<Vector3i> &getFaces() {return m_faces;}

    float getCubeSize() {return m_cube_size;}
    bool loadModel(string filename, int type = 1); // 1 - PLY, 2 - OBJ
    void savePLY(string filename);

    Vector3f bb_min,bb_max, centroid;
    Matrix<float,3,8> boundingBox;

    // The data of the model
    vector<Vector3f> m_normals, m_colors, m_points, m_localCoordColors;
    vector<Vector2f> m_tcoords;
    vector<Vector3i> m_faces;
    Mat m_tex;

    // Subsampled cloud for intermediate computations
    vector<Vector3f> m_subpoints, m_subnormals;
    vector<Vec3b> m_subcolors;
    
private:

    // Length of voxel [m] for subsampling.
    float m_cube_size;

    // Diameter of the object
    float m_diameter;

    // OpenGL specifics for fast rendering
    GLuint m_vbo_vertices, m_vbo_indices, m_vbo_tcoords, m_vbo_tex;
    vector<Vector3f> m_vertex;

}; 

}

