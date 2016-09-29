#include "bgfill.h"

namespace sz {

bgfill::bgfill()
{
}

void bgfill::loadBackgrounds(string backgrounds_path, int count /*=-1*/)
{
    filesystem::path dir(backgrounds_path);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << backgrounds_path << ". Aborting..." << endl;
        backgrounds = vector<Background>();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,6)=="color_")
                last = std::max(last,std::stoi(file.substr(6,file.length())));
        }
    if (count>-1) last = count;

    vector<Background> bgs;
    for (int i = 0; i <= last; i++)
    {
        Background bg;
        stringstream countf;
        countf << setw(4) << setfill('0') << to_string(i);
        bg.color = imread(backgrounds_path + "color_"  + countf.str() + ".png");
        bg.depth = imread(backgrounds_path + "depth_" + countf.str() + ".png",-1);
        assert(!bg.color.empty() && !bg.depth.empty());
        bg.depth.convertTo(bg.depth,CV_32F,0.001f);   // Bring depth map into meters
        // imshow("Color: ", bg.color);
        // imshow("Depth: ", bg.depth); waitKey();

        // Filter depth
        Mat depth_mini(bg.depth.size().height, bg.depth.size().width, CV_8UC1);
        bg.depth.convertTo(depth_mini, CV_8UC1, 255.0);
        resize(depth_mini, depth_mini, Size(), 0.2, 0.2);
        cv::inpaint(depth_mini, (depth_mini == 0.0), depth_mini, 5.0, INPAINT_TELEA);
        resize(depth_mini, depth_mini, bg.depth.size());
        depth_mini.convertTo(depth_mini, CV_32FC1, 1./255.0);
        depth_mini.copyTo(bg.depth, (bg.depth == 0));

        // Add normals
        depth2normals(bg.depth, bg.normals, 539, 539, 0, 0);
        // imshow("Normals: ", abs(bg.normals)); waitKey();
        // imshow("Depth: ", bg.depth); waitKey();

        // Scale backgrounds down
        Size bg_mini_size = bg.color.size()/3;
        resize(bg.color,bg.color,bg_mini_size);   // Standard bilinear interpolation
        resize(bg.normals,bg.normals,bg_mini_size);   // Standard bilinear interpolation
        resize(bg.depth,bg.depth,bg_mini_size,0,0,INTER_NEAREST); // Nearest-neighbor interpolation for depth!!!

        bgs.push_back(bg);
        loadbar("Loading backgrounds: ",i,last);
    }
    backgrounds = bgs;
}

void bgfill::randomRealFill(Mat &patch)
{
    if (backgrounds.empty()) throw runtime_error("No backgrounds loaded!");

    Size patch_size(patch.size().width,patch.size().height);
    Size bg_size(backgrounds[0].color.size().width, backgrounds[0].color.size().height);
    Mat tmp_rgb, tmp_dep, tmp_nor;

    // Split the patch
    vector<Mat> channels;
    cv::split(patch,channels);
    Mat patch_rgb,patch_dep,patch_nor;
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),patch_rgb);
    cv::merge(vector<Mat>({channels[3]}),patch_dep);
    cv::merge(vector<Mat>({channels[4],channels[5],channels[6]}),patch_nor);

    // Take random background
    std::uniform_int_distribution<int> r_bg(1, backgrounds.size()-1);
    std::uniform_int_distribution<int> r_x(patch_size.width/2, bg_size.width - patch_size.width/2);
    std::uniform_int_distribution<int> r_y(patch_size.height/2, bg_size.height - patch_size.height/2);

    // Find a center point
    int bg = r_bg(ran), center_x = r_x(ran), center_y = r_y(ran);

    // Check if image will be inside the bounds
    while( isnan(backgrounds[bg].depth.at<float>(center_x, center_y))
           || backgrounds[bg].depth.at<float>(center_x, center_y) < 0.4
           || backgrounds[bg].depth.at<float>(center_x, center_y) > 20)
        { bg = r_bg(ran), center_x = r_x(ran), center_y = r_y(ran); }

    int tl_x, tl_y; // Estimate top left corner
    tl_x = center_x - patch_size.width/2;
    tl_y = center_y - patch_size.height/2;

    backgrounds[bg].color(Rect(tl_x, tl_y, patch_size.width, patch_size.height)).copyTo(tmp_rgb);
    backgrounds[bg].depth(Rect(tl_x, tl_y, patch_size.width, patch_size.height)).copyTo(tmp_dep);
    backgrounds[bg].normals(Rect(tl_x, tl_y, patch_size.width, patch_size.height)).copyTo(tmp_nor);

    // Store the mask (dilated to kill render borders)
    Mat mask = patch_dep == 0;

    // Get outline mask using morphological gradient
    Mat dilate, erode;
    cv::dilate(mask,dilate,Mat());
    cv::erode(mask,erode,Mat());
    Mat outline = dilate - erode;

    // Adjust depth
    float depth_scale = 0.6 / backgrounds[bg].depth.at<float>(center_x, center_y);
    tmp_dep *= depth_scale;
    tmp_dep.setTo(1, tmp_dep > 1);

    // Fill backgrounds
    tmp_dep.copyTo(patch_dep,mask);
    tmp_nor.copyTo(patch_nor,mask);
    tmp_rgb.convertTo(tmp_rgb, CV_32FC3, 1/255.f);
    tmp_rgb.copyTo(patch_rgb,mask);

    // Smooth the edges
    Mat blurred_rgb, blurred_nor;
    medianBlur(patch_rgb, blurred_rgb, 3);
    medianBlur(patch_nor, blurred_nor, 3);
    blurred_rgb.copyTo(patch_rgb, outline);
    blurred_nor.copyTo(patch_nor, outline);

    cv::merge(vector<Mat>{patch_rgb,patch_dep,patch_nor},patch);
    // showRGBDPatch(patch, true);
}

void bgfill::randomColorFill(Mat &patch)
{
    int chans = patch.channels();
    std::uniform_real_distribution<float> p(0.f,1.f);

    vector<Mat> channels;
    cv::split(patch,channels);

    // Store the mask (dilated to kill render borders)
    Mat mask = (channels[3] == 0);
    cv::dilate(mask,mask,Mat());

    for (int r=0; r < patch.rows; ++r)
    {
        float *row = patch.ptr<float>(r);
        for (int c=0; c < patch.cols; ++c)
        {
            if (mask.at<uchar>(r,c))
                for (int ch = 0; ch < 7; ++ch)
                    row[c*chans + ch] =  p(ran);
        }
    }
}

void bgfill::randomShapeFill(Mat &patch)
{
    Size patch_size(patch.size().width,patch.size().height);

    // Split the patch
    vector<Mat> channels;
    cv::split(patch,channels);
    Mat patch_rgb,patch_dep,patch_nor;
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),patch_rgb);
    cv::merge(vector<Mat>({channels[3]}),patch_dep);
    cv::merge(vector<Mat>({channels[4],channels[5],channels[6]}),patch_nor);

    float scale_size = 1.2;
    // Store a copy and fill it with random shapes
    Mat tmp_rgb = Mat::zeros(patch_size.width*scale_size, patch_size.height*scale_size, CV_32FC3);
    Mat tmp_dep = Mat::zeros(patch_size.width*scale_size, patch_size.height*scale_size, CV_32F);
    Mat tmp_nor = Mat::zeros(patch_size.width*scale_size, patch_size.height*scale_size, CV_32FC3);
    Point center;

    std::uniform_real_distribution<float> color(0.35f,0.7f);
    std::uniform_real_distribution<float> s(0.0f,0.2f);
    std::uniform_int_distribution<int> r(0,20);
    std::vector<float> i{0, tmp_rgb.size().width/2-10.f, tmp_rgb.size().width/2+10.f, (float)tmp_rgb.size().width}; std::vector<float> w{1, 0, 0, 1};
    std::piecewise_linear_distribution<> coord(i.begin(), i.end(), w.begin());
    // std::uniform_int_distribution<int> coord(0,64);

    // Fill base surface
    float scale = s(ran);
    rectangle(tmp_rgb, Point(0,0), Point(tmp_rgb.cols, tmp_rgb.rows), Scalar(color(ran),color(ran),color(ran)), -1, 8);
    for (int y = 0; y < tmp_dep.size().height; ++y) {
        for (int x = 0; x < tmp_dep.size().width; ++x) {
             tmp_dep.at<float>(x,y) = 0.5 + scale * (float)x/(tmp_dep.size().width);
         }
     }

    // Fill circles
    for (int i = 0; i < 20; i++) {
      center.x = coord(ran);
      center.y = coord(ran);
      int rad = r(ran);
      circle(tmp_rgb, center, rad, Scalar(color(ran),color(ran),color(ran)), -1, 8);
      circle(tmp_dep, center, rad, Scalar(color(ran)), -1, 8);
    }

    // Adjust depth
    float depth_scale = 0.6 / tmp_dep.at<float>(tmp_dep.size().width/2, tmp_dep.size().height/2);
    tmp_dep *= depth_scale;
    tmp_dep.setTo(1, tmp_dep > 1);

    // Add noise
    Mat tmp_noise = tmp_dep.clone();
    randn(tmp_noise, 0.0f, 0.002f);
    tmp_dep += tmp_noise;

    depth2normals(tmp_dep, tmp_nor, 539, 539, 0, 0);

    // Store the mask (dilated to kill render borders)
    Mat mask = patch_dep == 0;
    cv::dilate(mask,mask,Mat());

    // Copy random shapes to the background of the patch
    tmp_rgb(Rect((tmp_rgb.cols - patch_size.width)/2, (tmp_rgb.rows - patch_size.width)/2, patch_size.width, patch_size.height)).copyTo(patch_rgb,mask);
    tmp_dep(Rect((tmp_dep.cols - patch_size.width)/2, (tmp_dep.cols - patch_size.width)/2, patch_size.width, patch_size.height)).copyTo(patch_dep,mask);
    tmp_nor(Rect((tmp_nor.cols - patch_size.width)/2, (tmp_nor.cols - patch_size.width)/2, patch_size.width, patch_size.height)).copyTo(patch_nor,mask);
    // cout << patch_dep << endl;

    cv::merge(vector<Mat>{patch_rgb,patch_dep,patch_nor},patch);
    // showRGBDPatch(patch, true);

}

void bgfill::randomFractalFill(Mat &patch)
{
    Size patch_size(patch.size().width,patch.size().height);

    // Split the patch
    vector<Mat> channels;
    cv::split(patch,channels);
    Mat patch_rgb,patch_dep,patch_nor;
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),patch_rgb);
    cv::merge(vector<Mat>({channels[3]}),patch_dep);
    cv::merge(vector<Mat>({channels[4],channels[5],channels[6]}),patch_nor);

    float scale_size = 1.2;
    // Store a copy and fill it with random shapes
    Mat tmp_rgb = Mat::zeros(patch_size.width*scale_size, patch_size.height*scale_size, CV_32FC3);
    Mat tmp_dep = Mat::zeros(patch_size.width*scale_size, patch_size.height*scale_size, CV_32F);
    Mat tmp_nor = Mat::zeros(patch_size.width*scale_size, patch_size.height*scale_size, CV_32FC3);

    FastNoise myNoise; // Create a FastNoise object
    myNoise.SetNoiseType(FastNoise::SimplexFractal); // Set the desired noise type
    myNoise.SetFrequency(0.01);
    myNoise.SetFractalGain(0.5);

    // Fill the noise (all channels)
    for (int c = 0; c < tmp_rgb.channels(); ++c) {
        myNoise.SetSeed(0);
        for (int y = 0; y < tmp_dep.rows; ++y) {
            for (int x = 0; x < tmp_dep.cols; ++x) {
                tmp_rgb.at<Vec3f>(x,y)[c] = myNoise.GetNoise(x,y) + 0.5f;
                if (tmp_rgb.at<Vec3f>(x,y)[c] < 0) tmp_rgb.at<Vec3f>(x,y)[c] = 0;
                if (tmp_rgb.at<Vec3f>(x,y)[c] > 1) tmp_rgb.at<Vec3f>(x,y)[c] = 1;
            }
        }
    }

    // Fill the depth noise
    for (int y = 0; y < tmp_dep.rows; ++y) {
        for (int x = 0; x < tmp_dep.cols; ++x) {
            tmp_dep.at<float>(x,y) = myNoise.GetNoise(x,y) * 0.5f + 1;
        }
    }

    // Adjust depth
    float depth_scale = 0.6 / tmp_dep.at<float>(tmp_dep.cols/2, tmp_dep.rows/2);
    tmp_dep *= depth_scale;
    tmp_dep.setTo(1, tmp_dep > 1);
    tmp_dep.setTo(0, tmp_dep < 0);

    // Store the mask
    Mat mask = patch_dep == 0;

    // Get outline mask using morphological gradient
    Mat dilate, erode;
    cv::dilate(mask,dilate,Mat());
    cv::erode(mask,erode,Mat());
    Mat outline = dilate - erode;

    depth2normals(tmp_dep, tmp_nor, 539, 539, 0, 0);

    tmp_rgb(Rect((tmp_rgb.cols - patch_size.width)/2, (tmp_rgb.rows - patch_size.width)/2, patch_size.width, patch_size.height)).copyTo(patch_rgb,mask);
    tmp_dep(Rect((tmp_dep.cols - patch_size.width)/2, (tmp_dep.cols - patch_size.width)/2, patch_size.width, patch_size.height)).copyTo(patch_dep,mask);
    tmp_nor(Rect((tmp_nor.cols - patch_size.width)/2, (tmp_nor.cols - patch_size.width)/2, patch_size.width, patch_size.height)).copyTo(patch_nor,mask);

    // Smooth the edges
    Mat blurred_rgb, blurred_nor;
    medianBlur(patch_rgb, blurred_rgb, 3);
    medianBlur(patch_nor, blurred_nor, 3);
    blurred_rgb.copyTo(patch_rgb, outline);
    blurred_nor.copyTo(patch_nor, outline);

    cv::merge(vector<Mat>{patch_rgb,patch_dep,patch_nor},patch);
}

}
