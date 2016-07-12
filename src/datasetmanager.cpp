#include "../include/datasetmanager.h"


namespace Gopnik
{

datasetManager::datasetManager(string config)
{
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);
    dataset_path = pt.get<string>("paths.dataset_path");
    hdf5_path = pt.get<string>("paths.hdf5_path");
    bg_path = pt.get<string>("paths.background_path");

    random_background = pt.get<bool>("input.random_background");
    use_real = pt.get<bool>("input.use_real");
    inplane = pt.get<bool>("input.inplane");
    models = to_array<string>(pt.get<string>("input.models"));
    used_models = to_array<string>(pt.get<string>("input.used_models"));
    rotInv = to_array<int>(pt.get<string>("input.rotInv"));
    nr_objects = used_models.size();

    // For each object build a mapping from model name to index number
    for (size_t i = 0; i < used_models.size(); ++i) model_index[used_models[i]] = i;
    // Global mapping
    for (size_t i = 0; i < models.size(); ++i) global_model_index[models[i]] = i;

}

Gopnik::Benchmark datasetManager::loadLinemodBenchmark(string linemod_path, string sequence, int count /*=-1*/)
{
    string dir_string = linemod_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Gopnik::Benchmark();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color")
                last = std::max(last,std::stoi(file.substr(5,file.length())));

        }

    if (count>-1) last = count;

    Gopnik::Benchmark bench;
    for (int i = 0; i <= last;i++)
    {
        Gopnik::Frame frame;
        frame.nr = i;
        frame.color = imread(dir_string + "/color" + to_string(i) + ".jpg");
        frame.depth = imread(dir_string + "/inp/depth" + to_string(i) + ".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);   // Bring depth map into meters
        ifstream pose(dir_string + "/pose" + to_string(i) + ".txt");
        assert(pose.is_open());

        frame.gt.push_back({sequence,Isometry3f::Identity()});

        for (int k=0; k < 4;k++)
            for(int l=0; l < 4;l++)
                pose >> frame.gt[0].second.matrix()(k,l);
        bench.frames.push_back(frame);

        loadbar("  - loading frames: ",i,last);
    }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;
    return bench;
}

Gopnik::Benchmark datasetManager::loadBigBirdBenchmark(string linemod_path, string sequence, int count /*=-1*/)
{
    string dir_string = linemod_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Gopnik::Benchmark();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,4)=="NP1_")
                last = std::max(last,std::stoi(file.substr(4,file.length())));
        }

    if (count > -1) last = count;

    Gopnik::Benchmark bench;

    // Read intristic cameras
    bench.cam = h5.readBBIntristicMats(dir_string + "/calibration.h5");

    // Read transformations
    vector<Isometry3f,Eigen::aligned_allocator<Isometry3f>> trans = h5.readBBTrans(dir_string + "/calibration.h5");

    for (int np = 1; np <= 5; ++np) {
        for (int i = 0; i <= last; i += 3)
        {
            Gopnik::Frame frame;
            frame.nr = i * np;
            frame.color = imread(dir_string + "/NP" + to_string(np) + "_" + to_string(i)+".jpg");
            frame.color = frame.color(Rect(0, 32, 1280, 960));
            resize(frame.color, frame.color, Size(640,480)); // rescale to 640*480
            // imshow("ColorScaled: ", frame.color); waitKey();
            frame.depth = h5.readBBDepth(dir_string + "/NP" + to_string(np) + "_" + to_string(i) + ".h5");
            // imshow("Depth: ", frame.depth); waitKey();
            assert(!frame.color.empty() && !frame.depth.empty());

            Isometry3f pose = h5.readBBPose(dir_string + "/poses/NP5" + "_" + to_string(i) + "_pose.h5");

            frame.gt.push_back({"object",Isometry3f::Identity()});

            frame.gt[0].second = trans[np-1] * pose.inverse();
            bench.frames.push_back(frame);
        }
        loadbar("  - loading frames: ", np, 5);
    }
    return bench;
}

vector<Background> datasetManager::loadBackgrounds(string backgrounds_path, int count /*=-1*/)
{

    filesystem::path dir(backgrounds_path);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << backgrounds_path << ". Aborting..." << endl;
        return vector<Background>();
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
//        imshow("Color: ", bg.color);
//        imshow("Depth: ", bg.depth); waitKey();

        // Filter depth
        Mat depth_mini(bg.depth.size().height, bg.depth.size().width, CV_8UC1);
        bg.depth.convertTo(depth_mini, CV_8UC1, 255.0);
        resize(depth_mini, depth_mini, Size(), 0.2, 0.2);
        cv::inpaint(depth_mini, (depth_mini == 0.0), depth_mini, 5.0, INPAINT_TELEA);
        resize(depth_mini, depth_mini, bg.depth.size());
        depth_mini.convertTo(depth_mini, CV_32FC1, 1./255.0);
        depth_mini.copyTo(bg.depth, (bg.depth == 0));
        // medianBlur(bg.depth, bg.depth, 5);

        // Add normals
        Gopnik::depth2normals(bg.depth, bg.normals, 539, 539, 0, 0);
//        imshow("Normals: ", abs(bg.normals)); waitKey();
//        imshow("Depth: ", bg.depth); waitKey();

        // Scale backgrounds down
        Size bg_mini_size = bg.color.size()/3;
        resize(bg.color,bg.color,bg_mini_size);   // Standard bilinear interpolation
        resize(bg.normals,bg.normals,bg_mini_size);   // Standard bilinear interpolation
        resize(bg.depth,bg.depth,bg_mini_size,0,0,INTER_NEAREST); // Nearest-neighbor interpolation for depth!!!

        bgs.push_back(bg);
        loadbar("  - loading backgrounds: ",i,last);
    }
    return bgs;
}

// Takes a color and depth frame and samples a normalized 4-channel patch at the given center position and z-scale
Mat datasetManager::samplePatchWithScale(Mat &color, Mat &depth, Mat &normals, int center_x, int center_y, float z, float fx, float fy)
{
    // Make a cut of metric size m
    float m = 0.2f;
    int screenW = fx * m/z;
    int screenH = fy * m/z;

    // Compute cut-out rectangle and ensure that we stay inside the image!
    Rect cut(center_x - screenW / 2, center_y - screenH / 2, screenW, screenH);
    if (cut.x < 0) cut.x = 0;
    if (cut.y < 0) cut.y = 0;
    if (cut.x > color.cols-screenW-1) cut.x = color.cols-screenW-1;
    if (cut.y > color.rows-screenH-1) cut.y = color.rows-screenH-1;

    assert (cut.x >= 0 && cut.x < color.cols-screenW);
    assert (cut.y >= 0 && cut.y < color.rows-screenH);

    // Cut-out from whole image
    Mat temp_col, temp_dep, temp_nor, final;
    color(cut).copyTo(temp_col);
    depth(cut).copyTo(temp_dep);
    normals(cut).copyTo(temp_nor);

    // Convert to float
    temp_col.convertTo(temp_col, CV_32FC3, 1/255.f);

    // Demean with central z value, clamp and rescale to [0,1]
    temp_dep -= z;
    temp_dep.setTo(-m, temp_dep < -m);
    temp_dep.setTo(m, temp_dep > m);
    temp_dep *= 1.0f / m;
    temp_dep = (temp_dep + 1.f) * 0.5f;

    // Resize
    const int CNN_INPUT_SIZE = 64;
    const Size final_size(CNN_INPUT_SIZE,CNN_INPUT_SIZE);
    resize(temp_col,temp_col,final_size);   // Standard bilinear interpolation
    resize(temp_nor,temp_nor,final_size);   // Standard bilinear interpolation
    resize(temp_dep,temp_dep,final_size,0,0,INTER_NEAREST);// Nearest-neighbor interpolation for depth!!!

    cv::merge(vector<Mat>{temp_col,temp_dep,temp_nor},final);

    return final;
}

vector<Gopnik::Sample> datasetManager::extractSceneSamplesPaul(vector<Gopnik::Frame> &frames, Matrix3f &cam, int index, Model &model)
{
    vector<Gopnik::Sample> samples;
    for (Gopnik::Frame &f : frames)
    {
//         Vector3f centroid = f.gt * model.centroid;
        Vector3f centroid = f.gt[0].second.translation();
        Vector3f projCentroid = cam * centroid;
        projCentroid /= projCentroid(2);

        // Compute normals
        Gopnik::depth2normals(f.depth,f.normals,cam);

        Gopnik::Sample s;
        s.data = samplePatchWithScale(f.color,f.depth,f.normals,projCentroid(0),projCentroid(1),centroid(2),cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion + translation
        s.label = Mat(1,8,CV_32F);
        s.label.at<float>(0,0) = index;
        Quaternionf q(f.gt[0].second.linear());
        for (int i=0; i < 4; ++i)
            s.label.at<float>(0,1+i) = q.coeffs()(i);
        for (size_t tr = 0; tr < 3; ++tr)
            s.label.at<float>(0,5+tr) = f.gt[0].second.inverse().translation()(tr);

        samples.push_back(s);

#if 0
        // Show rgbd patch
        showRGBDPatch(s.data);

        // Show centroid 2D
        Rect rect(projCentroid(0)-2, projCentroid(1)+2, 2, 2);
        rectangle(f.depth, rect, Scalar(0,0,255), 3);
        imshow("Depth", f.depth);

        // Visualize the cloud
        Mat cloud(f.depth.rows,f.depth.cols, CV_32FC3);
        depth2cloud(f.depth, cloud, cam);

        Mat object = viz::readCloud("/media/zsn/Storage/BMC/Master/Implementation/dataset_bigbird/detergent.ply");

        // Visualize for the case where feat_dim is 3D
        viz::Viz3d visualizer("Cloud");
        viz::WCloud wcloud(cloud, f.color);

        cv::Mat camcv; eigen2cv(f.gt.matrix(),camcv);
        visualizer.setViewerPose(cv::Affine3f::Identity());

        // visualizer.showWidget("coo", viz::WCoordinateSystem());
        visualizer.showWidget("cloud", wcloud); // cv::Affine3f::translation(cv::Vec3f(0,0,0))
        visualizer.showWidget("object", viz::WCloud(object, viz::Color::yellow()), cv::Affine3f(camcv));
        visualizer.showWidget("centroid", viz::WSphere(Point3f(centroid[0], centroid[1], centroid[2]), 0.02, 10, viz::Color::red()));
        visualizer.spin();

        // Render the model
        SphereRenderer renderer;
        renderer.init(cam);
        Mat col,dep;
        renderer.renderView(model,f.gt,col,dep,false);
#endif
    }
    return samples;
}

vector<Gopnik::Sample> datasetManager::extractSceneSamplesWadim(vector<Gopnik::Frame> &frames, Matrix3f &cam, int index)
{
    vector<Gopnik::Sample> samples;
    for (Gopnik::Frame &f : frames)
    {
        // Instead of taking object centroid, take the surface point as central sample point
        Vector3f projCentroid = cam * f.gt[0].second.translation();
        projCentroid /= projCentroid(2);

        float z = f.depth.at<float>(projCentroid(1),projCentroid(0));
        assert(z>0.0f);

        // Compute normals
        Gopnik::depth2normals(f.depth,f.normals,cam);

        Gopnik::Sample s;
        s.data = samplePatchWithScale(f.color,f.depth,f.normals,projCentroid(0),projCentroid(1),z,cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion + translation
        s.label = Mat(1,8,CV_32F);
        s.label.at<float>(0,0) = index;
        Quaternionf q(f.gt[0].second.linear());
        for (int i=0; i < 4; ++i)
            s.label.at<float>(0,1+i) = q.coeffs()(i);
        for (size_t tr = 0; tr < 3; ++tr)
            s.label.at<float>(0,5+tr) = f.gt[0].second.inverse().translation()(tr);

        samples.push_back(s);
    }
    return samples;
}

vector<Gopnik::Sample> datasetManager::createTemplatesPaul(Model &model, Matrix3f &cam, int index)
{
    ifstream file(dataset_path + "paul/camPositionsElAz.txt");
    assert(file.good());
    vector<Vector2f, Eigen::aligned_allocator<Vector2f> > sphereCoords(1542);

    // Read all camera poses from file (given in elevation and azimuth)
    vector<Matrix3f, Eigen::aligned_allocator<Matrix3f> > camPos;
    for(Vector2f &v : sphereCoords)
    {
        file >> v(0) >> v(1);

        // Copied from basetypes.py lines 100-xxx
        // To move into el=0,az=0 position we first have to rotate 45deg around x
        AngleAxisf camRot0(M_PI/2,Vector3f(1,0,0));

        // Build rotation matrix from spherical coordinates
        AngleAxisf el(v(0),Vector3f(1,0,0));
        AngleAxisf az(-v(1),Vector3f(0,0,1));
        Matrix3f camRot = (el*az).toRotationMatrix();
        camPos.push_back(camRot0*camRot);
    }

    // Render each and create proper sample
    SphereRenderer renderer;
    renderer.init(cam);
    Mat color,depth,normals;
    Isometry3f pose = Isometry3f::Identity();
    pose.translation()(2) = 0.4f;

    vector<Gopnik::Sample> samples;
    for (Matrix3f &m : camPos)
    {
        // Instead of taking object centroid, take the surface point as central sample point
        // float z = v.dep.at<float>(cam(1,2),cam(0,2));
        // assert(z>0.0f);
        float z = pose.translation()(2);

        pose.linear() = m;
        renderer.renderView(model,pose,color,depth,false);

        // Compute normals
        Gopnik::depth2normals(depth,normals,cam);

        Gopnik::Sample sample;
        sample.data = samplePatchWithScale(color,depth,normals,cam(0,2),cam(1,2),z,cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion + translation
        sample.label = Mat(1,8,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);
        for (size_t tr = 0; tr < 3; ++tr)
            sample.label.at<float>(0,5+tr) = pose.inverse().translation()(tr);

        samples.push_back(sample);
    }
    return samples;
}

vector<Gopnik::Sample> datasetManager::createTemplatesWadim(Model &model,Matrix3f &cam, int index, int subdiv)
{
    // Create synthetic views
    SphereRenderer sphere(cam);
    Mat normals;
    Vector3f scales(0.4, 1.1, 1.0);     // Render from 0.4 meters

    Vector3f in_plane_rots;
    if (inplane) { in_plane_rots = Vector3f(-45,15,45); }
            else { in_plane_rots = Vector3f(0,15,10); }

    vector<RenderView, Eigen::aligned_allocator<RenderView> > views =
            sphere.createViews(model,subdiv,scales,in_plane_rots,true,false);  // Equidistant sphere sampling with recursive level subdiv

    vector<Gopnik::Sample> samples;
    for (RenderView &v : views)
    {
        // Instead of taking object centroid, take the surface point as central sample point
        //        float z = v.dep.at<float>(cam(1,2),cam(0,2));
        //        assert(z>0.0f);
        float z = v.pose.translation()(2);

        Gopnik::depth2normals(v.dep,normals,cam);

        Gopnik::Sample sample;
        sample.data = samplePatchWithScale(v.col,v.dep,normals,cam(0,2),cam(1,2),z,cam(0,0),cam(1,1));

        // Build 6-dimensional label: model index + quaternion + translation
        sample.label = Mat(1,8,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(v.pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);

        for (size_t tr = 0; tr < 3; ++tr) {
            sample.label.at<float>(0,5+tr) = v.pose.inverse().translation()(tr);
        }

        samples.push_back(sample);
    }
    return samples;

}

void datasetManager::createSceneSamplesAndTemplates()
{
    for (size_t modelId = 0; modelId < used_models.size(); ++modelId) {

        string model_name = used_models[modelId];

        clog << "\nCreating samples and patches for " << model_name << ":" << endl;

        // - load model
        Model model;
        model.loadPLY(dataset_path + model_name + ".ply");

        // - load frames of benchmark and visualize
//         Benchmark bench = loadBigBirdBenchmark(dataset_path, model_name);
        Gopnik::Benchmark bench = loadLinemodBenchmark(dataset_path, model_name);
        // showGT(bench,model);

        // Real data
        // - for each scene frame, extract RGBD sample
        vector<Gopnik::Sample> realSamples = extractSceneSamplesPaul(bench.frames,bench.cam,model_index[model_name], model);

        // - store realSamples to HDF5 files
        h5.write(hdf5_path + "realSamples_" + model_name +".h5", realSamples);
        // for (Sample &s : realSamples) showRGBDPatch(s.data);

        // Synthetic data
        clog << "  - render synthetic data:" << endl;
        // - create synthetic samples and templates
        int subdivTmpl = 2; // sphere subdivision factor for templates
        vector<Gopnik::Sample> templates = createTemplatesWadim(model, bench.cam, model_index[model_name], subdivTmpl);
        vector<Gopnik::Sample> synthSamples = createTemplatesWadim(model, bench.cam, model_index[model_name], subdivTmpl+1);
//         vector<Sample> temp = createTemplatesPaul(model, bench.cam, model_index[model_name]);
//         vector<Sample> templates (temp.begin(),temp.begin() + 301);
//         vector<Sample> synthSamples (temp.begin() + 302, temp.end());

        // - store realSamples to HDF5 files
        h5.write(hdf5_path + "templates_" + model_name + ".h5", templates);
        h5.write(hdf5_path + "synthSamples_" + model_name + ".h5", synthSamples);
        // for (Sample &s : templates) showRGBDPatch(s.data);
//        for (Sample &s : synthSamples) showRGBDPatch(s.data);

    }
}

void datasetManager::generateDatasets()
{
    training_set.clear();
    test_set.clear();
    templates.clear();

    if (random_background)
        backgrounds = loadBackgrounds(bg_path);

    for (string &seq : used_models)
    {
        // Read the data from hdf5 files
        vector<Gopnik::Sample> train_real(h5.read(hdf5_path + "realSamples_" + seq + ".h5"));
        vector<Gopnik::Sample> train_synth(h5.read(hdf5_path + "synthSamples_" + seq + ".h5"));
        templates.push_back(h5.read(hdf5_path + "templates_" + seq + ".h5"));
        test_set.push_back(vector<Gopnik::Sample>());

//        if (random_background) {
//            for (Sample &s : train_synth) randomShapeFill(s.data);
//            for (Sample &s : templates.back()) randomShapeFill(s.data);
//            for (Sample &s : train_synth) randomColorFill(s.data);
//            for (Sample &s : templates.back()) randomColorFill(s.data);
//        }

        // Compute sizes and quaternions
        unsigned int nr_template_poses = templates[0].size();
        unsigned int nr_synth_poses = train_synth.size();
        unsigned int nr_real_poses = train_real.size();

        // - read quaternion poses from templates
        vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > tmpl_quats(nr_template_poses);
        for  (size_t i=0; i < nr_template_poses; ++i)
            for (size_t j=0; j < 4; ++j)
                tmpl_quats[i].coeffs()(j) = templates.back()[i].label.at<float>(0,1+j);

        // - read quaternion poses from synthetic data
        vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > synth_quats(nr_synth_poses);
        for (size_t i=0; i < nr_synth_poses; ++i)
            for (size_t j=0; j < 4; ++j)
                synth_quats[i].coeffs()(j) = train_synth[i].label.at<float>(0,1+j);

        // - read quaternion poses from real data
        vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > real_quats(nr_real_poses);
        for (size_t i=0; i < nr_real_poses; ++i)
            for (size_t j=0; j < 4; ++j)
                real_quats[i].coeffs()(j) = train_real[i].label.at<float>(0,1+j);

        // Find the closest templates for each real sample
        vector<vector<int>> maxSimTmpl(nr_template_poses, vector<int>());
        for (size_t real_sample = 0; real_sample < nr_real_poses; ++real_sample)
        {
            float best_dist = numeric_limits<float>::max();
            unsigned int sim_tmpl = 0;
            for (size_t tmpl = 0; tmpl < nr_template_poses; tmpl++)
            {
                float temp_dist = real_quats[real_sample].angularDistance(tmpl_quats[tmpl]);
                if (temp_dist >= best_dist) continue;
                best_dist = temp_dist;
                sim_tmpl = tmpl;
            }
            maxSimTmpl[sim_tmpl].push_back(real_sample);
        }

        // Divide between the training and test sets ~50/50
        for (size_t tmpl = 0; tmpl < nr_template_poses; ++tmpl) {
            if (!maxSimTmpl[tmpl].empty()) {
                // - to training set
                if (use_real) {
                    for (size_t i = 0; i < ceil(maxSimTmpl[tmpl].size()/2.0); ++i) {
                        train_synth.push_back(train_real[maxSimTmpl[tmpl][i]]);
                    }
                }
                // - to test set
                for (size_t i = ceil(maxSimTmpl[tmpl].size()/2.0); i < maxSimTmpl[tmpl].size(); ++i) {
                    test_set.back().push_back(train_real[maxSimTmpl[tmpl][i]]);
                }
            }
        }
        // Update the training set
        training_set.push_back(train_synth);
    }

    // Crop and shuffle the sets
    unsigned int min_training = numeric_limits<int>::max(), min_test = numeric_limits<int>::max();
    for (size_t object = 0; object < nr_objects; ++object) {
        if (min_training >= training_set[object].size()) min_training = training_set[object].size();
        if (min_test >= test_set[object].size()) min_test = test_set[object].size();
    }
    for (size_t object = 0; object < nr_objects; ++object) {
        training_set[object].resize(min_training);
        test_set[object].resize(min_test);
        random_shuffle(training_set[object].begin(), training_set[object].end());
        random_shuffle(test_set[object].begin(), test_set[object].end());
    }

    // Remember the sizes;
    nr_training_poses = training_set[0].size();
    nr_template_poses = templates[0].size();
    nr_test_poses = test_set[0].size();

    computeQuaternions();
}

void datasetManager::computeQuaternions()
{
    // Read quaternion poses from templates (they are identical for all objects)
    tmpl_quats.assign(nr_objects, vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>());
    for  (size_t i = 0; i < nr_objects; ++i) {
        tmpl_quats[i].resize(templates[i].size());
        for (size_t k = 0; k < tmpl_quats[i].size(); ++k)
            for (int j = 0; j < 4; ++j)
                tmpl_quats[i][k].coeffs()(j) = templates[i][k].label.at<float>(0,1+j);
    }

    // Read quaternion poses from training data
    training_quats.assign(nr_objects, vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>());
    for  (size_t i = 0; i < nr_objects; ++i) {
        training_quats[i].resize(training_set[i].size());
        for (size_t k = 0; k < training_quats[i].size(); ++k)
            for (int j = 0; j < 4; ++j)
                training_quats[i][k].coeffs()(j) = training_set[i][k].label.at<float>(0,1+j);
    }

    // Read quaternion poses from test data
    test_quats.assign(nr_objects, vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>());
    for  (size_t i = 0; i < nr_objects; ++i) {
        test_quats[i].resize(test_set[i].size());
        for (size_t k = 0; k < test_quats[i].size(); ++k)
            for (int j = 0; j < 4; ++j)
                test_quats[i][k].coeffs()(j) = test_set[i][k].label.at<float>(0,1+j);
    }
}

void datasetManager::randomColorFill(Mat &patch)
{
    int chans = patch.channels();
    std::uniform_real_distribution<float> p(0.f,1.f);
    for (int r=0; r < patch.rows; ++r)
    {
        float *row = patch.ptr<float>(r);
        for (int c=0; c < patch.cols; ++c)
        {
            if (row[c*chans + 3] > 0) continue;
            for (int ch = 0; ch < 7; ++ch)
                row[c*chans + ch] =  p(ran);
        }
    }
}

void datasetManager::randomShapeFill(Mat &patch)
{
    Size patch_size(patch.size().width,patch.size().height);

    // Split the patch
    vector<Mat> channels;
    cv::split(patch,channels);
    Mat patch_rgb,patch_dep,patch_nor;
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),patch_rgb);
    cv::merge(vector<Mat>({channels[3]}),patch_dep);
    cv::merge(vector<Mat>({channels[4],channels[5],channels[6]}),patch_nor);

    std::uniform_real_distribution<float> color(0.4f,0.8f);
    std::uniform_int_distribution<int> coord(0,64);
    std::uniform_int_distribution<int> r(0,40);

    // Store a copy and fill it with random shapes
    Mat tmp_rgb = Mat::zeros(patch_size.width, patch_size.height, CV_32FC3);
    Mat tmp_dep = Mat::zeros(patch_size.width, patch_size.height, CV_32F);
    for (int i = 0; i < 10; i++)
    {
      Point center;
      center.x = coord(ran);
      center.y = coord(ran);
      int rad = r(ran);
      circle(tmp_rgb, center, rad, Scalar(color(ran),color(ran),color(ran)), -1, 8);
      circle(tmp_dep, center, rad, Scalar(color(ran)), -1, 8);

    }
//    medianBlur(tmp_rgb, tmp_rgb, 5);
    GaussianBlur(tmp_rgb, tmp_rgb, Size(5,5), 0, 0);

    // Store the mask
    Mat mask = Mat::zeros(patch_size.width, patch_size.height, CV_8UC1);
    for (int y = 0; y < patch_size.height; ++y) {
        for (int x = 0; x < patch_size.width; ++x) {
            if (patch_dep.at<float>(x,y) > 0) mask.at<uchar>(x,y) = 255;
        }
    }

    // Copy random shapes to the background of the patch
    for (int y = 0; y < patch_size.height; ++y) {
        for (int x = 0; x < patch_size.width; ++x) {
            for (int c = 0; c < patch_rgb.channels(); ++c) {
                if (mask.at<uchar>(x,y) > 0) continue;
                patch_rgb.at<Vec3f>(x,y)[c] = tmp_rgb.at<Vec3f>(x,y)[c];
                patch_dep.at<float>(x,y) = tmp_dep.at<float>(x,y);
            }
        }
    }

    cv::merge(vector<Mat>{patch_rgb,patch_dep,patch_nor},patch);
//    showRGBDPatch(patch, true);

}

void datasetManager::randomBGFill(Mat &patch)
{
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
    float depth;
    int bg = r_bg(ran), center_x = r_x(ran), center_y = r_y(ran);

    // Check if image will be inside the bounds
    while( isnan(backgrounds[bg].depth.at<float>(center_x, center_y))
           || backgrounds[bg].depth.at<float>(center_x, center_y) < 0.4
           || backgrounds[bg].depth.at<float>(center_x, center_y) > 20)
        { bg = r_bg(ran), center_x = r_x(ran), center_y = r_y(ran); }

    int tl_x, tl_y; // Estimate top left corner
    depth = backgrounds[bg].depth.at<float>(center_x, center_y);
    tl_x = center_x - patch_size.width/2;
    tl_y = center_y - patch_size.height/2;

    tmp_rgb = backgrounds[bg].color(Rect(tl_x, tl_y, patch_size.width, patch_size.height));
    tmp_dep = backgrounds[bg].depth(Rect(tl_x, tl_y, patch_size.width, patch_size.height));
    tmp_nor = backgrounds[bg].normals(Rect(tl_x, tl_y, patch_size.width, patch_size.height));

    // Store the mask
    Mat mask = Mat::zeros(patch_size.width, patch_size.height, CV_8UC1);
    for (int y = 0; y < patch_size.height; ++y) {
        for (int x = 0; x < patch_size.width; ++x) {
            if (patch_dep.at<float>(x,y) > 0) mask.at<uchar>(x,y) = 255;
        }
    }

    // Adjust depth
    float depth_scale = 0.45 / backgrounds[bg].depth.at<float>(center_x, center_y);
    tmp_dep *= depth_scale;

    // Fill backgrounds
    tmp_rgb.convertTo(tmp_rgb, CV_32FC3, 1/255.f);
    for (int y = 0; y < patch_size.height; ++y) {
        for (int x = 0; x < patch_size.width; ++x) {
            for (int c = 0; c < patch_rgb.channels(); ++c) {
                if (mask.at<uchar>(x,y) > 0) continue;
                patch_rgb.at<Vec3f>(x,y)[c] = tmp_rgb.at<Vec3f>(x,y)[c]; // RGB
                patch_dep.at<float>(x,y) = tmp_dep.at<float>(x,y); // Depth
                patch_nor.at<Vec3f>(x,y)[c] = tmp_nor.at<Vec3f>(x,y)[c]; // Normals
            }
        }
    }
    medianBlur(patch_rgb, patch_rgb, 3);
    medianBlur(patch_nor, patch_nor, 3);

    cv::merge(vector<Mat>{patch_rgb,patch_dep,patch_nor},patch);
//    showRGBDPatch(patch, true);

}

}
