

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
#include <unordered_map>

#include <H5Cpp.h>

#define SQR(a) ((a)*(a))

/*

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

pair<Mat,Mat> loadPCLCloud(string filename)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::io::loadPCDFile(filename,cloud);
    Mat col(cloud.height,cloud.width,CV_8UC3),out(cloud.height,cloud.width,CV_32FC3);
    for (int r = 0; r < out.rows; ++r)
        for (int c = 0; c < out.cols; ++c)
        {
            pcl::PointXYZRGB p = cloud.at(c,r);
            int color = *reinterpret_cast<int*>(&p.rgb);
            col.at<Vec3b>(r,c) = Vec3b(color&255,(color>>8)&255,(color>>16)&255);
            out.at<Vector3f>(r,c) = Vector3f(p.x,p.y,p.z);
        }
    return {col,out};
}

Benchmark loadWillow(string folder)
{
    filesystem::directory_iterator end_iter;
    std::unordered_map< string, vector<string> > scenes;
    for(filesystem::directory_iterator dir_iter(folder + "/scenes"); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_directory(dir_iter->status()))
        {
            string dir = dir_iter->path().leaf().string();
            //if (dir.find("willow")!=string::npos)
            {
                for(filesystem::directory_iterator dir_iter2(dir_iter->path()); dir_iter2 != end_iter ; ++dir_iter2)
                {
                    string file = dir_iter2->path().leaf().string();
                    if (file.find(".pcd")!=string::npos) scenes[dir].push_back(file.substr(0,file.size()-4));

                }
            }
        }

    vector<string> sorted_seqs;
    for (auto &e : scenes) sorted_seqs.push_back(e.first);
    std::sort(sorted_seqs.begin(),sorted_seqs.end());

    vector<Frame> frames;
    for (string &seq : sorted_seqs)
    {
        //cerr << seq << endl;
        std::sort(scenes[seq].begin(),scenes[seq].end());
        for (size_t frame = 0; frame < scenes[seq].size(); frame++)
        {
            //cerr << scenes[seq][frame] << endl;
            Frame fr;
            fr.nr = frames.size();
            pair<Mat,Mat> data = loadPCLCloud(folder + "/scenes/"+seq+"/"+scenes[seq][frame]+".pcd");
            fr.color = data.first;
            fr.cloud = data.second;
            vector<Mat> chans;
            cv::split(data.second,chans);
            fr.depth = chans[2];
            fr.depth.setTo(0, fr.depth != fr.depth);    //NaNs to 0

            Mat temp;
            fr.depth.convertTo(temp,CV_16U,1000);
            imwrite(folder + "/scenes/"+seq+"/"+scenes[seq][frame]+"_color.png",fr.color);
            imwrite(folder + "/scenes/"+seq+"/"+scenes[seq][frame]+"_depth.png",temp);

            string filter = to_string(frame);
            while (filter.size() < 10) filter = "0" + filter;
            filter = "cloud_" + filter + "_";
            for(filesystem::directory_iterator dir_iter(folder + "/ground_truth/" + seq); dir_iter != end_iter ; ++dir_iter)
                if (filesystem::is_regular_file(dir_iter->status()))
                {
                    string file = dir_iter->path().leaf().string();
                    if (file.find(filter)==string::npos) continue;
                    if (file.find(".txt")==string::npos) continue;
                    if (file.find("occlusion")!=string::npos) continue;
                    ifstream gt(dir_iter->path().string());
                    Instance inst;
                    inst.first = file.substr(17,9); // extract object name
                    for(int i=0; i < 4; ++i)
                        for(int j=0; j < 4; ++j)
                            gt >> inst.second.matrix()(i,j);
                    fr.gt.push_back(inst);
                }
            frames.push_back(fr);
        }
    }

    // Enumerate all models and load them
    set<string> models;
    for (Frame &f : frames)
        for (Instance &i : f.gt)
            models.insert(i.first);

    Benchmark bench;
    bench.frames = frames;
    for (string s : models) bench.models.push_back({folder + "/models/"+s+"_mesh.ply",s});

#if 0
    for (string s : models)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
        pcl::io::loadPCDFile(folder + "/models/"+s+".pcd",cloud);
        Model m,m2;
        for (auto &p : cloud.points) m.getPoints().push_back(Vector3f(p.x,p.y,p.z));
        for (auto &p : cloud.points) m.getColors().push_back(Vector3f(p.r/255.0,p.g/255.0,p.b/255.0));
        for (auto &p : cloud.points) m.getNormals().push_back(Vector3f(p.normal[0],p.normal[1],p.normal[2]));
        m.savePLY(folder + "/models/"+s+".ply");
        m2.loadPLY(folder + "/models/"+s+"_mesh.ply");
        for (size_t i=0; i < m2.getPoints().size(); ++i)
        {
            Vector3f &p = m2.getPoints()[i];
            size_t best=0;
            float best_dist=1000000;
            for (size_t j=0; j < m.getPoints().size(); ++j)
            {
                float dist = (m2.getPoints()[i]-m.getPoints()[j]).squaredNorm();
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best = j;
                }
            }
            m2.getColors()[i] = m.getColors()[best];
        }
        m2.savePLY(folder + "/models/"+s+"_mesh.ply");
    }
#endif

    // Only approx. values
    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 525.0f;
    bench.cam(0,2) = 319.5f;
    bench.cam(1,1) = 525.0f;
    bench.cam(1,2) = 239.5f;

    return bench;

}
*/



/*
pcl::PointCloud<pcl::PointXYZ>::Ptr CV2PCL(Mat &cloud)
{
    // WARNING: PCL has R,C the other way. We basicallly compute on a transposed image!!
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl_cloud->width = cloud.rows;
    pcl_cloud->height = cloud.cols;
    pcl_cloud->points.resize(pcl_cloud->width*pcl_cloud->height);
    for (int r = 0; r < cloud.rows; ++r)
        for (int c = 0; c < cloud.cols; ++c)
            pcl_cloud->at(r,c).getVector3fMap() = cloud.at<Vector3f>(r,c);
    return pcl_cloud;
}

Mat PCL2CV(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    Mat out(cloud->width,cloud->height,CV_32FC3);
    for (int r = 0; r < out.rows; ++r)
        for (int c = 0; c < out.cols; ++c)
                out.at<Vector3f>(r,c) = cloud->at(r,c).getNormalVector3fMap();
    return out;
}
*/


/*
void convertLinemodBenchmark(string dir_string)
{

    // This functions loads the original Linemod benchmark
    // and converts into meter, centered and up-rotated object model

    cerr << "Converting " << dir_string << "..." << endl;
    filesystem::path dir(dir_string + "/data");
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        exit(0);
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="depth")
            {
                int number = atoi(file.substr(5,file.length()).c_str());
                last = max(last,number);
            }
        }

    detectionTUM::DetectionObject obj;
    // obj.loadModel(dir_string + "/OLDmesh.ply",false);
    obj.loadModel(dir_string + "/object.xyz",false);

    Isometry3f transform = Isometry3f::Identity();
    if (filesystem::exists(filesystem::path(dir_string + "/transform.dat")))
    {
        ifstream transform_dat(dir_string + "/transform.dat");
        int trash;
        transform_dat >> trash;
        for (int i=0; i < 12; i++)
        {
            transform_dat >> trash;
            transform_dat >> transform.matrix()(i/4,i%4);
        }
    }

    // Bring from mm into meters and position onto local center
    for (Vector3f &p : obj.model.getPoints()) p = transform*(p*0.001f);

    // Demean cloud
    Vector3f centroid(0,0,0);
    for (Vector3f &p : obj.model.getPoints()) centroid += p;
    centroid /= obj.model.getPoints().size();
    for (Vector3f &p : obj.model.getPoints()) p -= centroid;

    // Rotate 180° around x-axis
    AngleAxisf xrot(M_PI,Vector3f(1,0,0));
    for (Vector3f &p : obj.model.getPoints())
        p = xrot*p;

    obj.model.savePLY(dir_string + "/NEWmesh.ply");

    for (int i=0; i <= last;i++)
    {
        ostringstream ss;
        ss << i;

        ifstream file(dir_string + "/data/depth"+ss.str()+".dpt",ofstream::in|ofstream::binary);
        if(!file.is_open()) continue;
        int row,col;
        file.read((char*)&row,sizeof(int));
        file.read((char*)&col,sizeof(int));
        Mat depth(row,col,CV_16U);
        for(int r=0;r<row;++r)
            for(int c=0;c<col;++c)
                file.read((char*)&depth.at<ushort>(r,c),sizeof(ushort));
        file.close();
        if(!depth.empty()) imwrite(dir_string + "/data/depth"+ss.str()+".png",depth);

        Isometry3f pose = Isometry3f::Identity();
        ifstream rot(dir_string + "/data/rot"+ss.str()+".rot");
        rot >> pose.linear()(0,0) >> pose.linear()(0,0);
        for (int k=0; k < 3;k++)
            for(int l=0; l < 3;l++)
                rot >> pose.linear()(k,l);
        ifstream tra(dir_string + "/data/tra"+ss.str()+".tra");
        tra >> pose.translation()(0) >> pose.translation()(0);
        for (int k=0; k < 3;k++) tra >> pose.translation()(k);
        pose.translation() *= 0.01f;    // Translation is in cm

        // Subtract mean from ground truth pose and rotate around X
        Isometry3f pose_inv = pose.inverse();
        pose_inv.translation() -= centroid;
        pose_inv = xrot*pose_inv;
        pose = pose_inv.inverse();

        ofstream output(dir_string + "/data/pose"+ss.str()+".txt");
        output << pose.matrix() << endl;

    }
}*/

using namespace boost;
using namespace std;

namespace Gopnik
{

pair<Mat,Mat> loadPCLCloud(string filename)
{
    assert(0);

    /*
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::io::loadPCDFile(filename,cloud);
    Mat col(cloud.height,cloud.width,CV_8UC3),out(cloud.height,cloud.width,CV_32FC3);
    for (int r = 0; r < out.rows; ++r)
        for (int c = 0; c < out.cols; ++c)
        {
            pcl::PointXYZRGB p = cloud.at(c,r);
            int color = *reinterpret_cast<int*>(&p.rgb);
            col.at<Vec3b>(r,c) = Vec3b(color&255,(color>>8)&255,(color>>16)&255);
            out.at<Vector3f>(r,c) = Vector3f(p.x,p.y,p.z);
        }
    return {col,out};
    */

    return {Mat(),Mat()};

}



Benchmark loadKinectBenchmark(string folder)
{

    filesystem::directory_iterator end_iter;
    vector<string> scenes;
    for(filesystem::directory_iterator dir_iter(folder + "/scenes"); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()))
        {
            string file = dir_iter->path().leaf().string();
            if (file.find(".pcd")!=string::npos) scenes.push_back(file);
        }
    std::sort(scenes.begin(),scenes.end());

    Benchmark bench;
    for (string &seq : scenes)
    {
        Frame fr;
        fr.nr = bench.frames.size();
        pair<Mat,Mat> data = loadPCLCloud(folder + "/scenes/"+seq);
        fr.color = data.first;
        vector<Mat> chans;
        cv::split(data.second,chans);
        fr.depth = chans[2];
        fr.depth.setTo(0, fr.depth != fr.depth);    //NaNs to 0
        string filter = seq.substr(0,seq.size()-4);
        for(filesystem::directory_iterator dir_iter(folder + "/gt_or_format"); dir_iter != end_iter ; ++dir_iter)
            if (filesystem::is_regular_file(dir_iter->status()))
            {
                string file = dir_iter->path().leaf().string();
                if (file.find(filter)==string::npos) continue;
                if (file.find(".txt")==string::npos) continue;
                if (file.find("occlusion")!=string::npos) continue;
                ifstream gt(dir_iter->path().string());
                Instance inst;
                inst.first = file.substr(filter.size()+1,file.size()-filter.size()-1+-6); // extract object name
                cerr << inst.first << endl;
                for(int i=0; i < 4; ++i)
                    for(int j=0; j < 4; ++j)
                        gt >> inst.second.matrix()(i,j);
                fr.gt.push_back(inst);
            }
        bench.frames.push_back(fr);
    }

    for(filesystem::directory_iterator dir_iter(folder + "/models"); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.find(".icp")!=string::npos) continue;
            if (file.find(".ply")!=string::npos) bench.models.push_back({folder + "/models/"+file,file.substr(0,file.size()-4)});
        }

    return bench;


}


Benchmark loadJHUBenchmark(string folder)
{


    Benchmark bench;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(folder + "/mesh"); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.find(".icp")!=string::npos) continue;
            if (file.find(".ply")!=string::npos) bench.models.push_back({folder + "/mesh/"+file,file.substr(0,file.size()-4)});
        }

    std::unordered_map< string, vector<string> > scenes;
    for(filesystem::directory_iterator dir_iter(folder + "/scene"); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_directory(dir_iter->status()))
        {
            string dir = dir_iter->path().leaf().string();
            for(filesystem::directory_iterator dir_iter2(dir_iter->path()); dir_iter2 != end_iter ; ++dir_iter2)
            {
                string file = dir_iter2->path().leaf().string();
                if (file.find(".pcd")!=string::npos) scenes[dir].push_back(file);
            }

        }

    vector<string> sorted_seqs;
    for (auto &e : scenes) sorted_seqs.push_back(e.first);
    std::sort(sorted_seqs.begin(),sorted_seqs.end());

    sorted_seqs.resize(1);

    for (string &seq : sorted_seqs)
    {
        std::sort(scenes[seq].begin(),scenes[seq].end());
        for (size_t frame = 0; frame < scenes[seq].size(); frame++)
        {
            Frame fr;
            fr.nr = bench.frames.size();
            pair<Mat,Mat> data = loadPCLCloud(folder + "/scene/"+seq+"/"+scenes[seq][frame]);
            fr.color = data.first;
            vector<Mat> chans;
            cv::split(data.second,chans);
            fr.depth = chans[2];
            fr.depth.setTo(0, fr.depth != fr.depth);    //NaNs to 0

            string filename = scenes[seq][frame];
            size_t pos = filename.find_last_of("_");
            int frame_id = stoi(filename.substr(pos+1,filename.size()-pos-5));

            for(filesystem::directory_iterator dir_iter(folder + "/scene/"+seq+"/poses"); dir_iter != end_iter ; ++dir_iter)
                if (filesystem::is_regular_file(dir_iter->status()))
                {
                    string file = dir_iter->path().leaf().string();
                    if (file.find(".csv")==string::npos) continue;

                    size_t us = file.find_last_of("_");
                    size_t end = file.find_last_of(".");
                    string model = file.substr(0,us);
                    int id = stoi(file.substr(us+1,end-us-1));
                    if (id != frame_id) continue;

                    ifstream gt(dir_iter->path().string());
                    int num;
                    gt >> num;
                    for(int i = 0 ; i < num ; i++)
                    {
                        Instance inst;
                        inst.first = model;
                        Vector3f t;
                        gt >> t(0) >> t(1) >> t(2);
                        float x,y,z,w;
                        gt >> x >> y >> z >> w;
                        inst.second = Isometry3f::Identity();
                        inst.second.translation() = t;
                        inst.second.linear() =  Eigen::Quaternionf(w,x,y,z).toRotationMatrix();
                        fr.gt.push_back(inst);
                    }
                }
            bench.frames.push_back(fr);
        }
    }

    return bench;

}

Benchmark loadWillowBenchmark(string folder)
{

    Benchmark bench;

    // Only approx. values
    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = bench.cam(1,1) = 525.0f;
    bench.cam(1,2) = 319.5f;
    bench.cam(0,2) = 239.5f;

    std::swap(bench.cam(1,2),bench.cam(0,2));

    string dir_string = folder + "/wadim";
    int last = 339;
    cout << "Loading frames in the range " << 0 << " - " << last << endl;
    for (int nr=0; nr <= last;nr++)
    {
        Frame frame;
        frame.nr = nr;
        string nr_string = to_string(nr);
        frame.color = imread(dir_string + "/color"+nr_string+".png");
        frame.depth = imread(dir_string + "/depth"+nr_string+".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);
        ifstream pose(dir_string + "/poses"+nr_string+".txt");
        assert(pose.is_open());
        int nr_gts;
        pose >> nr_gts;
        for (int i=0; i < nr_gts; ++i)
        {
            Instance inst;
            pose >> inst.first;
            for (int k=0; k < 4;k++)
                for(int l=0; l < 4;l++)
                    pose >> inst.second.matrix()(k,l);

#if 1
                    inst.second.linear() = AngleAxisf(M_PI/2, Vector3f::UnitY()).toRotationMatrix() *
                            AngleAxisf(M_PI/2, Vector3f::UnitZ()).toRotationMatrix() *
                            inst.second.linear();
                    inst.second.linear() = Matrix3f::Identity();
                    Vector3f gt_tr = inst.second.translation();
                    Point p = cloud2depth(gt_tr,bench.cam);
                    std::swap(p.x,p.y);
                    p.y = (640-480)+(480-p.y);
                    inst.second.translation() = depth2cloud(p,gt_tr(2),bench.cam);
#endif

            frame.gt.push_back(inst);
        }


#if 1
            cv::transpose(frame.color,frame.color);
            cv::flip(frame.color,frame.color,0);
            cv::transpose(frame.depth,frame.depth);
            cv::flip(frame.depth,frame.depth,0);
#endif

        bench.frames.push_back(frame);
    }

    // Enumerate all models and load them
    set<string> models;
    for (Frame &f : bench.frames)
        for (Instance &i : f.gt)
            models.insert(i.first);

    for (string s : models) bench.models.push_back({folder + "/models/"+s+"_mesh.ply",s});

    /*
        SphereRenderer renderer;
        Matrix3f temp;
        temp << 525,0,320,0,525,240,0,0,1;
        renderer.init(temp);
        Mat ren_c,ren_d;
        for (DetectionObject &obj : DB.objects)
        {
            Isometry3f iso = Isometry3f::Identity();
            iso.linear() = (AngleAxisf(M_PI/5,Vector3f::UnitZ())*AngleAxisf(M_PI/1.5,Vector3f::UnitX())).toRotationMatrix();
            iso.translation()(2) = 0.5f;
            renderer.renderView(obj,iso,ren_c,ren_d,false);
            imwrite(obj.name+".png",ren_c);
        }
    */


    return bench;

}


Benchmark loadWillowBenchmarkOLD(string folder)
{

    Benchmark bench;

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = bench.cam(1,1) = 525.0f;
    bench.cam(0,2) = 319.5f;
    bench.cam(1,2) = 239.5f;


    //std::swap(bench.cam(1,2),bench.cam(0,2));


    filesystem::directory_iterator end_iter;
    std::unordered_map< string, vector<string> > scenes;
    for(filesystem::directory_iterator dir_iter(folder + "/scenes"); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_directory(dir_iter->status()))
        {
            string dir = dir_iter->path().leaf().string();
            if (dir.find("willow")!=string::npos) continue;
            //{
                for(filesystem::directory_iterator dir_iter2(dir_iter->path()); dir_iter2 != end_iter ; ++dir_iter2)
                {
                    string file = dir_iter2->path().leaf().string();
                    if (file.find("color.png")!=string::npos) scenes[dir].push_back(file.substr(0,file.size()-9));

                }
            //}
        }

    vector<string> sorted_seqs;
    for (auto &e : scenes) sorted_seqs.push_back(e.first);
    std::sort(sorted_seqs.begin(),sorted_seqs.end());

    for (string &seq : sorted_seqs)
    {
        //cerr << seq << endl;
        std::sort(scenes[seq].begin(),scenes[seq].end());
        for (size_t frame = 0; frame < scenes[seq].size(); frame++)
        {
            //cerr << scenes[seq][frame] << endl;
            Frame fr;
            fr.nr = bench.frames.size();
            fr.color = imread(folder + "/scenes/"+seq+"/"+scenes[seq][frame]+"color.png");
            fr.depth = imread(folder + "/scenes/"+seq+"/"+scenes[seq][frame]+"depth.png",-1);
            assert(!fr.color.empty() && !fr.depth.empty());
            fr.depth.convertTo(fr.depth,CV_32F,0.001f);

#if 0
            cv::transpose(fr.color,fr.color);
            cv::flip(fr.color,fr.color,0);
            cv::transpose(fr.depth,fr.depth);
            cv::flip(fr.depth,fr.depth,0);
#endif
            string filter = to_string(frame);
            while (filter.size() < 10) filter = "0" + filter;
            filter = "cloud_" + filter + "_";
            for(filesystem::directory_iterator dir_iter(folder + "/ground_truth/" + seq); dir_iter != end_iter ; ++dir_iter)
                if (filesystem::is_regular_file(dir_iter->status()))
                {
                    string file = dir_iter->path().leaf().string();
                    if (file.find(filter)==string::npos) continue;
                    if (file.find(".txt")==string::npos) continue;
                    if (file.find("occlusion")!=string::npos) continue;
                    ifstream gt(dir_iter->path().string());
                    Instance inst;
                    inst.first = file.substr(17,9); // extract object name
                    for(int i=0; i < 4; ++i)
                        for(int j=0; j < 4; ++j)
                            gt >> inst.second.matrix()(i,j);

#if 0
                    inst.second.linear() = AngleAxisf(M_PI/2, Vector3f::UnitY()).toRotationMatrix() *
                            AngleAxisf(M_PI/2, Vector3f::UnitZ()).toRotationMatrix() *
                            inst.second.linear();
                    inst.second.linear() = Matrix3f::Identity();
                    Vector3f gt_tr = inst.second.translation();
                    Point p = cloud2depth(gt_tr,old_cam);
                    std::swap(p.x,p.y);
                    p.y = (640-480)+(480-p.y);
                    inst.second.translation() = depth2cloud(p,gt_tr(2),bench.cam);
#endif
                    fr.gt.push_back(inst);
                }
            bench.frames.push_back(fr);
        }
    }

    // Enumerate all models and load them
    set<string> models;
    for (Frame &f : bench.frames)
        for (Instance &i : f.gt)
            models.insert(i.first);

    for (string s : models) bench.models.push_back({folder + "/models/"+s+"_mesh.ply",s});

#if 0
    for (string s : models)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
        pcl::io::loadPCDFile(folder + "/models/"+s+".pcd",cloud);
        Model m,m2;
        for (auto &p : cloud.points) m.getPoints().push_back(Vector3f(p.x,p.y,p.z));
        for (auto &p : cloud.points) m.getColors().push_back(Vector3f(p.r/255.0,p.g/255.0,p.b/255.0));
        for (auto &p : cloud.points) m.getNormals().push_back(Vector3f(p.normal[0],p.normal[1],p.normal[2]));
        m.savePLY(folder + "/models/"+s+".ply");
        m2.loadPLY(folder + "/models/"+s+"_mesh.ply");
        for (size_t i=0; i < m2.getPoints().size(); ++i)
        {
            Vector3f &p = m2.getPoints()[i];
            size_t best=0;
            float best_dist=1000000;
            for (size_t j=0; j < m.getPoints().size(); ++j)
            {
                float dist = (m2.getPoints()[i]-m.getPoints()[j]).squaredNorm();
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best = j;
                }
            }
            m2.getColors()[i] = m.getColors()[best];
        }
        m2.savePLY(folder + "/models/"+s+"_mesh.ply");
    }
#endif


#if 0 // rebuild groundtruth
    for (auto &s : bench.models) DB.loadObject(s.first,s.second,true);
    for (Frame &f : bench.frames)
    {
        imwrite("color"+to_string(f.nr)+".png",f.color);
        Mat temp;
        f.depth.convertTo(temp,CV_16U,1000);
        imwrite("depth"+to_string(f.nr)+".png",temp);
        ofstream p("poses"+to_string(f.nr)+".txt");
        p << f.gt.size() << endl;
        for (Instance &inst : f.gt)
        {
            Isometry3f pose_inv = inst.second.inverse();
            pose_inv.translation() -= DB.getObject(inst.first)->old_centroid;
            inst.second = pose_inv.inverse();
            p << inst.first << endl << inst.second.matrix() << endl;
        }
    }
#endif


    return bench;

}



Benchmark loadExportBenchmark(string dir_string, int count)
{

    Benchmark bench;
    cerr << "Loading export benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color")
                last = max(last,stoi(file.substr(6,file.length())));
        }

    if (count>0) last = count;
    cout << "Loading frames in the range " << 0 << " - " << last << endl;
    int counter=0;
    for (int nr=50; nr <= last;nr++)
    {
        Frame frame;
        frame.nr = nr;
        string nr_string = to_string(nr);
        while (nr_string.size() < 6) nr_string = '0' + nr_string;
        frame.nr = counter++;
        frame.color = imread(dir_string + "/color_"+nr_string+".png");
        frame.depth = imread(dir_string + "/depth_"+nr_string+".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);
        bench.frames.push_back(frame);
    }


    set<string> models;
    for (Frame &f : bench.frames)
        for (Instance &i : f.gt)
            models.insert(i.first);

   // models.insert("17_Pet_bottle_pet_tea");

    for (string s : models) bench.models.push_back({"/home/kehl/GIT/work/3Dmodels/ply_file_bin/"+s+".ply",s});

    //bench.models.push_back({dir_string + "/14_MugCup_green.ply","14_MugCup_green"});
    bench.models.push_back({dir_string + "/17_Pet_bottle_pet_tea.ply","17_Pet_bottle_pet_tea"});
    //bench.models.push_back({dir_string + "/head.ply","head"});


    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 539.81f;
    bench.cam(0,2) = 318.27f;
    bench.cam(1,1) = 539.83f;
    bench.cam(1,2) = 239.56f;

    return bench;

}

Benchmark loadToyotaBenchmark(string dir_string, int count)
{

    Benchmark bench;
    cerr << "Loading Toyota benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color")
                last = max(last,stoi(file.substr(6,file.length())));
        }

    if (count>0) last = count;
    cout << "Loading frames in the range " << 0 << " - " << last << endl;
    for (int nr=0; nr <= last;nr++)
    {
        Frame frame;
        frame.nr = nr;
        string nr_string = to_string(nr);
        while (nr_string.size() < 6) nr_string = '0' + nr_string;
        frame.color = imread(dir_string + "/color_"+nr_string+".png");
        frame.depth = imread(dir_string + "/depth_"+nr_string+".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);
        ifstream pose(dir_string + "/groundtruth_"+nr_string+".txt");
        assert(pose.is_open());
        int nr_gts;
        pose >> nr_gts;
        for (int i=0; i < nr_gts; ++i)
        {
            Instance inst;
            pose >> inst.first;
            for (int k=0; k < 4;k++)
                for(int l=0; l < 4;l++)
                    pose >> inst.second.matrix()(k,l);
            frame.gt.push_back(inst);
        }
        bench.frames.push_back(frame);
    }

    set<string> models;
    for (Frame &f : bench.frames)
        for (Instance &i : f.gt)
            models.insert(i.first);

    for (string s : models) bench.models.push_back({"/home/kehl/GIT/3Dmodels/"+s+".ply",s});

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 539.81f;
    bench.cam(0,2) = 318.27f;
    bench.cam(1,1) = 539.83f;
    bench.cam(1,2) = 239.56f;

    return bench;

}

Benchmark loadTejaniBenchmark(string seq)
{
    string tejani_path = "/home/kehl/TEJANI/";

    string dir_string = tejani_path + seq;

    Benchmark bench;
    cerr << "Loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cerr << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return bench;
    }

    vector<string> indices;
    filesystem::path color_path(dir_string+"/RGB/");
    filesystem::path depth_path(dir_string+"/Depth/");

    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(color_path); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(file.size()-3)!="png") continue;
            if (file.length()==11) indices.push_back((file.substr(4,3)));
            else indices.push_back((file.substr(4,4)));
        }

    //indices.resize(20);

    cout << "Loading " << indices.size() << " frames" << endl;

    ifstream off(dir_string + "/offset.txt");
    assert(off.is_open());
    Vector3f offset;
    off >> offset(0) >> offset(1) >> offset(2);
    int frame_counter = 0;
    for (string i : indices)
    {
        Frame frame;
        frame.nr = stoi(i); //frame_counter++;
        frame.color = imread(color_path.string() + "img_" + i + ".png");
        frame.depth = imread(depth_path.string() + "img_" + i + ".png",-1);

        assert(!frame.color.empty() && !frame.depth.empty());

        frame.depth.convertTo(frame.depth,CV_32F,0.001f); //millimeter to meter

        ifstream poses(dir_string + "/Annotation/" + "img_" + i + ".txt");
        assert(poses.is_open());
        int insts;
        poses >> insts;
        char t;
        for (int ins=0; ins < insts; ++ins)
        {
            Instance gt;
            gt.first = "model";

            for (int m=0; m < 4; ++m)
            {
                poses >> gt.second.matrix()(m,0) >> t;
                poses >> gt.second.matrix()(m,1) >> t;
                poses >> gt.second.matrix()(m,2) >> t;
                poses >> gt.second.matrix()(m,3);
            }
            gt.second.translation() *= 0.001;

            // Adjust pose s.t. we have upper-z and demeaned
            Isometry3f inv = gt.second.inverse();
            inv.translation() += offset;
            inv = AngleAxisf(M_PI,Vector3f(1,0,0))*inv;
            gt.second = inv.inverse();

            frame.gt.push_back(gt);
        }
        bench.frames.push_back(frame);
    }

    bench.models.push_back({tejani_path+seq+"/model.ply","model"});

    bench.cam.setIdentity();
    bench.cam(0,0) = 571.9737f;
    bench.cam(0,2) = 319.5000f;
    bench.cam(1,1) = 571.0073;
    bench.cam(1,2) = 239.5000f;

    return bench;
}

Benchmark loadTejaniBenchmarkCorrected(string seq)
{
    string tejani_path = "/home/kehl/TEJANI/";

    string dir_string = tejani_path + seq;

    Benchmark bench;
    cerr << "Loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cerr << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return bench;
    }

    filesystem::path poses_path(dir_string+"/corrected");
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(poses_path); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)!="poses") continue;

            Frame fr;
            fr.nr = std::stoi(file.substr(5,file.length()));
            ifstream poses(dir_string+"/corrected/"+file);
            assert(poses.is_open());
            int insts;
            poses >> insts;
            for (int ins=0; ins < insts; ++ins)
            {
                Isometry3f pose;
                for (int m=0; m < 4; ++m)
                    for (int n=0; n < 4; ++n)
                        poses >> pose.matrix()(m,n);
                fr.gt.push_back({seq,pose});
            }
            bench.frames.push_back(fr);

        }

    int lol=0;
    for (Frame &f : bench.frames)
    {
        string nr = to_string(f.nr);
        while (nr.size() < 3) nr = "0" + nr;
        f.color = imread(dir_string+"/RGB/"     + "img_" + nr + ".png");
        f.depth = imread(dir_string+"/Depth/"   + "img_" + nr + ".png",-1);
        assert(!f.color.empty() && !f.depth.empty());
        f.depth.convertTo(f.depth,CV_32F,0.001f); //millimeter to meter
        f.nr = lol++;
    }


    bench.models.push_back({tejani_path+seq+"/model_corrected.ply",seq});

    bench.cam.setIdentity();
    bench.cam(0,0) = 571.9737f;
    bench.cam(0,2) = 319.5000f;
    bench.cam(1,1) = 571.0073;
    bench.cam(1,2) = 239.5000f;

    return bench;
}

Benchmark loadTLESSBenchmark(string seq)
{
    string tless_path = "/home/kehl/Dropbox/TLESS/";

    string dir_string = tless_path + seq;


    // Hodan is'n Hoden. We have to map weird object ids to their model number...
    std::unordered_map<int,int> hoden_map;
    hoden_map[100] = 8;
    hoden_map[200] = 9;

    hoden_map[101] = 2;
    hoden_map[201] = 4;

    Benchmark bench;

    cerr << "Loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return bench;
    }
    int last=0;

    filesystem::path color_path(dir_string+"/rgb_primesense/");
    filesystem::path depth_path(dir_string+"/depth_primesense/");

    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(color_path); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(file.size()-3)=="png")
                last = std::max(last,std::stoi(file.substr(0,4)));
        }

    cout << "Loading frames in the range " << 0 << " - " << last << endl;
    ifstream poses(dir_string + "/gt.txt");
    assert(poses.is_open());

    for (int i=0; i <= last;i++)
    {
        Frame frame;
        frame.nr = i;
        string nr = std::to_string(i);
        while (nr.size() < 4) nr.insert(nr.begin(),'0');
        frame.color = imread(color_path.string() + nr + ".png");
        frame.depth = imread(depth_path.string() + nr + ".png",-1);

        assert(!frame.color.empty() && !frame.depth.empty());

        frame.depth.convertTo(frame.depth,CV_32F,0.0001f); //micrometer to meter

        cv::resize(frame.color,frame.color,Size(640,480));

        int in_frame,in_count,in_id;
        poses >> in_frame >> in_count;
        assert(in_frame == frame.nr);
        for (int i=0; i < in_count; ++i)
        {
            Instance gt;

            poses >> in_id;
            assert(hoden_map.find(in_id)!= hoden_map.end());
            gt.first = std::to_string(hoden_map[in_id]);

            Vector3f rot,tra;
            poses >> rot(0) >> rot(1) >> rot(2) >> tra(0) >> tra(1) >> tra(2);
            gt.second = Isometry3f::Identity();
            gt.second.linear() =
                    (AngleAxisf(rot(0), Vector3f::UnitX())
                     * AngleAxisf(rot(1),Vector3f::UnitY())
                     * AngleAxisf(rot(2),Vector3f::UnitZ())).toRotationMatrix();
            gt.second.translation() = tra*0.001;

            //if (gt.first == "2") continue;
            if (gt.first == "4") continue;
            if (gt.first == "8") continue;
            if (gt.first == "9") continue;

            //cerr << gt.first << endl << gt.second.matrix() << endl;

            frame.gt.push_back(gt);
        }

        bench.frames.push_back(frame);
    }

    set<string> models;
    for (Frame &f : bench.frames)
        for (Instance &i : f.gt)
            models.insert(i.first);

    bench.models.push_back({tless_path+seq+"/model_bin.ply",seq});

    bench.cam.setIdentity();
    return bench;
}


Benchmark loadKoboBenchmark(string dir_string)
{
    cerr << "Loading benchmark " << dir_string << endl;
    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }

    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color") last = std::max(last,stoi(file.substr(6)));
        }

    cout << "Loading frames in the range " << 0  << " - " << last << endl;
    Benchmark bench;
    for (int i=0; i <= last;i++)
    {
        Frame frame;
        frame.nr = i;
        string nr = std::to_string(i);
        while (nr.size() < 6) nr.insert(nr.begin(),'0');
        frame.color = imread(dir_string + "/color_"+nr+".png");
        frame.depth = imread(dir_string + "/depth_"+nr+".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());



        Mat depthf, temp, temp2;
        frame.depth.convertTo(depthf, CV_8UC1, 255.0/2048.0);

        // 1 step - downsize for performance, use a smaller version of depth image
       // Mat small_depthf;
       // resize(depthf, small_depthf, Size(), 0.2, 0.2);

        // 2 step - inpaint only the masked "unknown" pixels
        cv::inpaint(depthf,depthf ==0, temp, 5.0, INPAINT_TELEA);

        // 3 step - upscale to original size and replace inpainted regions in original depth image
        //resize(temp, temp2, depthf.size());
        temp.copyTo(depthf, depthf == 0); // add to the original signal
        depthf.convertTo(frame.depth, CV_16UC1, 2048.0/255.0);



        frame.depth.convertTo(frame.depth,CV_32F,0.001f);
        //imshow("test",frame.depth);waitKey();

        bench.frames.push_back(frame);
    }

    // Collect models and groundtruth
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_directory(dir_iter->status()) )
        {
            vector<string> gt_files;
            string model = dir_iter->path().leaf().string();
            bench.models.push_back({dir_string+"/"+model +"/"+model+".ply",model});
            for(filesystem::directory_iterator dir_iter2(dir_iter->path()); dir_iter2 != end_iter ; ++dir_iter2)
            {
                string file = dir_iter2->path().leaf().string();
                if (file.substr(0,11)=="groundtruth") gt_files.push_back(file);
            }
            std::sort(gt_files.begin(),gt_files.end());
            assert(gt_files.size()==bench.frames.size());
            for (size_t i=0; i < bench.frames.size();i++)
            {
                ifstream pose(dir_string + "/" + model + "/" + gt_files[i]);
                assert(pose.is_open());
                Instance inst;
                inst.first = model;
                for (int k=0; k < 4;k++)
                    for(int l=0; l < 4;l++)
                        pose >> inst.second.matrix()(k,l);
                bench.frames[i].gt.push_back(inst);
            }
        }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 539.81f;
    bench.cam(0,2) = 318.27f;
    bench.cam(1,1) = 539.83f;
    bench.cam(1,2) = 239.56f;
    return bench;
}

Benchmark loadLinemodBenchmark(string seq, int count, int start)
{


#ifdef __APPLE__
    string LINEMOD_path = "/Users/kehl/Dropbox/LINEMOD/";
#else
    string LINEMOD_path = "/home/kehl/Dropbox/LINEMOD/";
#endif



    string dir_string = LINEMOD_path + seq;
    cerr << "Loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color")
                last = max(last,stoi(file.substr(5,file.length())));
        }

    if (count>0) last = count;
    cout << "Loading frames in the range " << start  << " - " << last << endl;
    Benchmark bench;
    for (int i=start; i <= last;i++)
    {
        Frame frame;
        frame.nr = i;
        frame.color = imread(dir_string + "/color"+to_string(i)+".jpg");
        frame.depth = imread(dir_string + "/inp/depth"+to_string(i)+".png",-1);
        //frame.depth = imread(dir_string + "/depth"+to_string(i)+".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);
        ifstream pose(dir_string + "/pose"+to_string(i)+".txt");
        assert(pose.is_open());
        Instance inst;
        inst.first = seq;
        for (int k=0; k < 4;k++)
            for(int l=0; l < 4;l++)
                pose >> inst.second.matrix()(k,l);
        frame.gt.push_back(inst);
        bench.frames.push_back(frame);
    }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;

    bench.models.push_back({LINEMOD_path+seq+".ply",seq});
    return bench;
}

Benchmark loadLinemodOcclusionBenchmark(string seq, int count)
{


#ifdef __APPLE__
    string LINEMOD_path = "/Users/kehl/Dropbox/LINEMOD/";
#else
    string LINEMOD_path = "/home/kehl/Dropbox/LINEMOD/";
    //string LINEMOD_path = "/mnt/sdb1/LINEMOD/";
#endif

    string dir_string = LINEMOD_path + "occlusion/";
    cerr << "Loading occlusion benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }

    Benchmark bench;

    int last = 1024;
    if (count > 0) last = count;
    for (int i=0; i < last; ++i)
    {
        Frame f;
        f.nr = i;
        string nr = to_string(f.nr), temp;
        while (nr.size() < 5) nr = "0" + nr;
        f.color = imread(dir_string + "/rgb_noseg/color_"+nr+".png");
        f.depth = imread(dir_string + "/depth_noseg/depth_"+nr+".png",-1);
        assert(!f.color.empty() && !f.depth.empty());
        f.depth.convertTo(f.depth,CV_32F,0.001f);

        ifstream pose(dir_string + seq + "/info_"+nr+".txt");
        assert(pose.is_open());
        Instance inst;
        inst.first = seq;
        inst.second = Isometry3f::Identity();
        pose >> temp >> temp >> temp >> temp >> temp >> temp;
        for (int k=0; k < 3;k++)
            for(int l=0; l < 3;l++)
                pose >> inst.second.matrix()(k,l);
        pose >> temp;
        for (int k=0; k < 3;k++) pose >> inst.second.translation()(k);
        inst.second.translation()(2) *= -1;
        f.gt.push_back(inst);
        bench.frames.push_back(f);
    }
    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;

    bench.models.clear();
    bench.models.push_back({dir_string+seq+".ply",seq});
    return bench;
}



void writeHDF5(string filename, vector<Sample> &samples)
{
    if (samples.empty())
    {
        cerr << "writeHDF5: Nothing to write!" << endl;
        return;
    }

    try
    {
        // The HDF5 data layer in Caffe expects the following (everything float!):
        // Data must be Samples x Channels x Height x Width
        // Labels must be Samples x FeatureDim
        vector<hsize_t> data_dims = {samples.size(),(hsize_t) samples[0].data.channels(),(hsize_t) samples[0].data.rows,(hsize_t) samples[0].data.cols};

        // Specify size and shape of subset to write. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> slab_size = {1, data_dims[1], data_dims[2],data_dims[3]};

        H5::DataSpace data_space(data_dims.size(),data_dims.data());
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DataSet data_set = file.createDataSet("data",H5::PredType::NATIVE_FLOAT,data_space);

        // Fill temporary memory how Caffe needs it by filling the correct hyperslab per sample in the HDF5
        vector<float> temp(data_dims[1]*data_dims[2]*data_dims[3]);
        for(uint i=0; i < samples.size(); ++i)
        {
            for (hsize_t ch = 0; ch < data_dims[1]; ++ch)
                for (hsize_t y = 0; y < data_dims[2]; ++y)
                    for (hsize_t x = 0; x < data_dims[3]; ++x)
                        temp[ch*data_dims[2]*data_dims[3] + y*data_dims[3] + x] = samples[i].data.ptr<float>(y)[x*data_dims[1] + ch];

            offset[0] = i;
            data_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            data_set.write(temp.data(), H5::PredType::NATIVE_FLOAT, memspace, data_space);
        }

        vector<hsize_t> label_dims = {samples.size(), (hsize_t) samples[0].label.cols};
        H5::DataSpace label_space(label_dims.size(),label_dims.data());
        H5::DataSet label_set = file.createDataSet("label",H5::PredType::NATIVE_FLOAT,label_space);
        slab_size = {1, label_dims[1]};
        memspace = H5::DataSpace(slab_size.size(), slab_size.data());
        for(uint i=0; i < samples.size(); ++i)
        {
            offset[0] = i;
            label_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            label_set.write(samples[i].label.data, H5::PredType::NATIVE_FLOAT, memspace, label_space);
        }

    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}


void writeHDF5TensorFlow(string filename, vector<Sample> &samples)
{
    if (samples.empty())
    {
        cerr << "writeHDF5: Nothing to write!" << endl;
        return;
    }
    try
    {
        Sample &s = samples[0];
        vector<hsize_t> d_dims = {samples.size(),(hsize_t)s.data.cols,(hsize_t)s.data.rows,(hsize_t)s.data.channels()};
        vector<hsize_t> l_dims = {samples.size(),(hsize_t)s.label.cols,(hsize_t)s.label.rows,(hsize_t)s.label.channels()};

        // Specify size and shape of subset to write. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> d_slab_size = {1, d_dims[1], d_dims[2], d_dims[3]};
        vector<hsize_t> l_slab_size = {1, l_dims[1], l_dims[2], l_dims[3]};

        H5::DataSpace d_space(d_dims.size(),d_dims.data());
        H5::DataSpace d_mem(d_slab_size.size(), d_slab_size.data());
        H5::DataSpace l_space(l_dims.size(),l_dims.data());
        H5::DataSpace l_mem(l_slab_size.size(), l_slab_size.data());

        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DataSet d_set = file.createDataSet("data",H5::PredType::NATIVE_FLOAT,d_space);
        H5::DataSet l_set = file.createDataSet("label",H5::PredType::NATIVE_FLOAT,l_space);
        for(uint i=0; i < samples.size(); ++i)
        {
            offset[0] = i;
            d_space.selectHyperslab(H5S_SELECT_SET, d_slab_size.data(),offset.data());
            d_set.write(samples[i].data.data, H5::PredType::NATIVE_FLOAT, d_mem, d_space);
            l_space.selectHyperslab(H5S_SELECT_SET, l_slab_size.data(),offset.data());
            l_set.write(samples[i].label.data, H5::PredType::NATIVE_FLOAT, l_mem, l_space);
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}

vector<Sample> readHDF5TensorFlow(string filename, int count)
{
    vector<Sample> samples;
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet d_set = file.openDataSet("data");
        H5::DataSet l_set = file.openDataSet("label");

        H5::DataSpace d_space = d_set.getSpace();
        H5::DataSpace l_space = l_set.getSpace();

        vector<hsize_t> d_dims(d_space.getSimpleExtentNdims());
        vector<hsize_t> l_dims(l_space.getSimpleExtentNdims());
        d_space.getSimpleExtentDims(d_dims.data(), nullptr);
        l_space.getSimpleExtentDims(l_dims.data(), nullptr);

        assert(d_dims[0] == l_dims[0]); // Make sure that data count = label count

        // Specify size and shape of subset to read. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> d_slab_size = {1, d_dims[1], d_dims[2],d_dims[3]};
        vector<hsize_t> l_slab_size = {1, l_dims[1], l_dims[2],l_dims[3]};

        H5::DataSpace d_mem(d_slab_size.size(), d_slab_size.data());
        H5::DataSpace l_mem(l_slab_size.size(), l_slab_size.data());

        samples.resize(d_dims[0]);
        if (count>0) samples.resize(count);
        for(uint i=0; i < samples.size(); ++i)
        {
            offset[0] = i;
            samples[i].data =  Mat(d_dims[1],d_dims[2],CV_32FC(d_dims[3]));
            samples[i].label = Mat(l_dims[1],l_dims[2],CV_32FC(l_dims[3]));
            d_space.selectHyperslab(H5S_SELECT_SET, d_slab_size.data(),offset.data());
            d_set.read(samples[i].data.data, H5::PredType::NATIVE_FLOAT, d_mem, d_space);
            l_space.selectHyperslab(H5S_SELECT_SET, l_slab_size.data(),offset.data());
            l_set.read(samples[i].label.data, H5::PredType::NATIVE_FLOAT, l_mem, l_space);
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
    return samples;
}


vector<Sample> readHDF5(string filename, int start, int count)
{
    vector<Sample> samples;
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet data_set = file.openDataSet("data");
        H5::DataSet label_set = file.openDataSet("label");

        H5::DataSpace data_space = data_set.getSpace();
        H5::DataSpace label_space = label_set.getSpace();

        vector<hsize_t> data_dims(data_space.getSimpleExtentNdims());
        vector<hsize_t> label_dims(label_space.getSimpleExtentNdims());
        data_space.getSimpleExtentDims(data_dims.data(), nullptr);
        label_space.getSimpleExtentDims(label_dims.data(), nullptr);

        assert(data_dims[0] == label_dims[0]); // Make sure that data count = label count

        // Specify size and shape of subset to read. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> slab_size = {1, data_dims[1], data_dims[2],data_dims[3]};
        H5::DataSpace memspace(slab_size.size(), slab_size.data());


        int nr_samples = count > -1 ? count : data_dims[0];

        // Copy from Caffe layout back into OpenCV layout
        vector<float> temp(data_dims[1]*data_dims[2]*data_dims[3]);   // Memory for one patch
        samples.resize(nr_samples);
        for (uint i=0; i < samples.size(); ++i)
        {
            // Select correct patch as hyperslab inside the HDF5 file
            offset[0] = start + i;
            data_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            data_set.read(temp.data(), H5::PredType::NATIVE_FLOAT, memspace, data_space);

            samples[i].data = Mat(data_dims[2],data_dims[3],CV_32FC(data_dims[1]));
            for (hsize_t chan=0; chan < data_dims[1]; ++chan)
            {
                auto currChan = chan*data_dims[2]*data_dims[3];
                for (hsize_t y =0; y < data_dims[2]; ++y)
                    for (hsize_t x =0; x < data_dims[3]; ++x)
                        samples[i].data.ptr<float>(y)[x*data_dims[1] + chan] = temp[currChan + y*data_dims[3] + x];
            }
        }

        slab_size = {1, label_dims[1]};
        memspace = H5::DataSpace(slab_size.size(), slab_size.data());
        for (uint i=0; i < samples.size(); ++i)
        {
            // Select correct label as hyperslab inside the HDF5 file
            samples[i].label = Mat(1,label_dims[1],CV_32F);
            offset[0] = start + i;
            label_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            label_set.read(samples[i].label.data, H5::PredType::NATIVE_FLOAT, memspace, label_space);
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
    return samples;
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

Mat maskPlaneRANSAC(Mat &cloud, Mat &normals)
{
    assert(!cloud.empty() && !normals.empty());
    default_random_engine gen;

    Mat mask = Mat::zeros(cloud.size(),CV_8U);


    Vector3f *cloud_array = (Vector3f*) cloud.data;
    Vector3f *normal_array = (Vector3f*) normals.data;
    uniform_int_distribution<int> ran_index(0,cloud.cols*cloud.rows-1);

    vector<Vector3f> pts(3), nors(3);

    Vector3f best_plane;
    float best_off;
    int best_inliers=0;
    int trials=0;
    while(true)
    {
        if (trials>20) break;

        // Draw 3 unique points
        pts.clear();
        nors.clear();
        for(int i=0; i<500; ++i)
        {
            int idx = ran_index(gen);
            if (cloud_array[idx](2)==0) continue;
            pts.push_back(cloud_array[idx]);
            nors.push_back(normal_array[idx]);
            if (pts.size()==3) break;
        }
        if (pts.size() < 3) break;  // Depth map possibly zeros

        if (nors[0].dot(nors[1]) < 0.9) continue;
        if (nors[0].dot(nors[2]) < 0.9) continue;

        trials++;
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

}
