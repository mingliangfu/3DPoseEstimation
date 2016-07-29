#include "../include/datasetmanager.h"


namespace sz {

Benchmark datasetManager::loadLinemodBenchmark(string linemod_path, string sequence, int count /*=-1*/)
{
    string dir_string = linemod_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }
    int last=0;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()))
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color")
                last = std::max(last,std::stoi(file.substr(5,file.length())));
        }

    if (count>-1) last = count;

    Benchmark bench;
    for (int i = 0; i <= last;i++)
    {
        Frame frame;
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

        loadbar("  - loading frames: ", i, last);
    }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;
    return bench;
}

Benchmark datasetManager::loadBigbirdBenchmark(string bigbird_path, string sequence, int count /*=-1*/)
{
    string dir_string = bigbird_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;

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
            if (file.substr(0,4)=="NP1_")
                last = std::max(last,std::stoi(file.substr(4,file.length())));
        }

    if (count > -1) last = count;

    Benchmark bench;

    // Read intristic cameras
    bench.cam = h5.readBBIntristicMats(dir_string + "/calibration.h5");

    // Read transformations
    vector<Isometry3f,Eigen::aligned_allocator<Isometry3f>> trans = h5.readBBTrans(dir_string + "/calibration.h5");

    for (int np = 1 ; np <= 5; ++np) {
        for (int i = 0; i <= last; i += 3)
        {
            Frame frame;
            frame.nr = i * np;
            frame.color = imread(dir_string + "/NP" + to_string(np) + "_" + to_string(i)+".jpg");
            resize(frame.color, frame.color, Size(640,512)); // rescale to 640*512
            // imshow("ColorScaled: ", frame.color); waitKey();
            frame.depth = Mat::zeros(frame.color.size(),CV_32F);
            Mat demo = h5.readBBDepth(dir_string + "/NP" + to_string(np) + "_" + to_string(i) + ".h5");
            resize(demo, demo, Size(576,432));
            demo.copyTo(frame.depth(Rect(27, 35, demo.cols, demo.rows)));
            // imshow("Depth: ", frame.depth); waitKey();
            assert(!frame.color.empty() && !frame.depth.empty());

            Isometry3f pose = h5.readBBPose(dir_string + "/poses/NP5" + "_" + to_string(i) + "_pose.h5");
            frame.gt.push_back({"object", Isometry3f::Identity()});
            frame.gt[0].second = trans[np-1] * pose.inverse();
            bench.frames.push_back(frame);
        }
        loadbar("  - loading frames: ", np, 5);
    }
    return bench;
}

Benchmark datasetManager::loadBenjaminBenchmark(string benjamin_path, string sequence, int index)
{
    string dir_string = benjamin_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }
    int last=0;
    Benchmark bench;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter)
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            Frame frame;
            string file = dir_iter->path().leaf().string();
            if (file.find("_color.png")< file.size())
            {
                // Read color
                frame.color = imread(dir_string + "/" + file);

                // Read depth from binary
                ifstream fileStream(dir_string + "/" + file.substr(0, file.find("_color.png")) + "_depth.raw", ios::binary);
                uint16_t m, rows, cols;
                fileStream.read((char*)&rows, sizeof(uint16_t));
                fileStream.read((char*)&cols, sizeof(uint16_t));

                frame.depth = Mat::zeros(rows, cols, CV_32FC1); //Matrix to store values

                for (int i = 0; i < rows*cols; ++i) {
                    fileStream.read((char*)&m, sizeof(uint16_t));
                    int temprow = i / cols;
                    int tempcol = i % cols;
                    frame.depth.at<float>(temprow, tempcol) = (float)m*0.0001f;
                    if (m != 0) cout << (float)m*0.0001f << endl;
                }

                frame.color.convertTo(frame.color,CV_32FC3, 1/255.f);

                imshow("Color: ", frame.color);
                imshow("Depth: ", frame.depth); waitKey();

                // Add pose matrix
                ifstream pose(dir_string + "/" + file.substr(0, file.find("_color.png")) + "_pose.txt");
                assert(pose.is_open());

                frame.gt.push_back({sequence,Isometry3f::Identity()});

                for (int k=0; k < 4;k++)
                    for(int l=0; l < 4;l++)
                        pose >> frame.gt[0].second.matrix()(k,l);

                bench.frames.push_back(frame);

//                loadbar("  - loading frames: ", i, last);
            }
        }

        // Save intristic matrix
        bench.cam = Matrix3f::Identity();
        bench.cam(0,0) = 572.4114f;
        bench.cam(0,2) = 325.2611f;
        bench.cam(1,1) = 573.5704f;
        bench.cam(1,2) = 242.0489f;
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
        // medianBlur(bg.depth, bg.depth, 5);

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
    return bgs;
}

// Takes a color and depth frame and samples a normalized 4-channel patch at the given center position and z-scale
Mat datasetManager::samplePatchWithScale(Mat &color, Mat &depth, Mat &normals, int center_x, int center_y, float z, float fx, float fy)
{
    // Make a cut of metric size m
    float m;
    if (dataset_name == "LineMOD") {m = 0.2f;}
    else if (dataset_name == "BigBIRD") {m = 0.25f;}
    else if (dataset_name == "Washington") {m = 0.2f;}
    else {m = 0.2f;}
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

vector<Sample> datasetManager::extractRealSamplesPaul(vector<Frame> &frames, Matrix3f &cam, int index, Model &model)
{
    vector<Sample> samples;
    for (Frame &f : frames)
    {
        Vector3f centroid;
        if (dataset_name == "LineMOD") {centroid = f.gt[0].second.translation();}
        else if (dataset_name == "BigBIRD") {centroid = f.gt[0].second * model.centroid;}
        else if (dataset_name == "Washington") {centroid = f.gt[0].second.translation();}

        Vector3f projCentroid = cam * centroid;
        projCentroid /= projCentroid(2);

        // Compute normals
        depth2normals(f.depth,f.normals,cam);

        Sample sample;
        sample.data = samplePatchWithScale(f.color,f.depth,f.normals,projCentroid(0),projCentroid(1),centroid(2),cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion + translation
        sample.label = Mat(1,8,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(f.gt[0].second.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);
        for (size_t tr = 0; tr < 3; ++tr)
            sample.label.at<float>(0,5+tr) = f.gt[0].second.inverse().translation()(tr);

        samples.push_back(sample);
    }
    return samples;
}

vector<Sample> datasetManager::extractRealSamplesWadim(vector<Frame> &frames, Matrix3f &cam, int index)
{
    vector<Sample> samples;
    for (Frame &f : frames)
    {
        // Instead of taking object centroid, take the surface point as central sample point
        Vector3f projCentroid = cam * f.gt[0].second.translation();
        projCentroid /= projCentroid(2);

        float z = f.depth.at<float>(projCentroid(1),projCentroid(0));
        assert(z>0.0f);

        // Compute normals
        depth2normals(f.depth,f.normals,cam);

        Sample sample;
        sample.data = samplePatchWithScale(f.color,f.depth,f.normals,projCentroid(0),projCentroid(1),z,cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion + translation
        sample.label = Mat(1,8,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(f.gt[0].second.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);
        for (size_t tr = 0; tr < 3; ++tr)
            sample.label.at<float>(0,5+tr) = f.gt[0].second.inverse().translation()(tr);

        samples.push_back(sample);
    }
    return samples;
}

vector<Sample> datasetManager::createSynthSamplesPaul(Model &model, Matrix3f &cam, int index)
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
    Mat color, depth, normals;
    Isometry3f pose = Isometry3f::Identity();
    Vector3f trans;
    if (dataset_name == "LineMOD") {trans = Vector3f(0,0,0.4);}
    else if (dataset_name == "BigBIRD") {trans = Vector3f(0,0,0.72);}
    else if (dataset_name == "Washington") {trans = Vector3f(0,0,0.4);}
    else {trans = Vector3f(0,0,0.4);}

    vector<Sample> samples;
    for (Matrix3f &m : camPos)
    {
        // Take object centroid
        pose.linear() = m; pose.translation() = trans;

        if (dataset_name == "BigBIRD") // adapt the pose
        {
            // Vector3f shift = pose.linear() * model.centroid + trans;
            // pose.translation() = trans + (trans - shift);
            Isometry3f pose_object = pose.inverse();
            pose_object.translation() += model.centroid;
            pose = pose_object.inverse();
        }

        // Render the view
        renderer.renderView(model,pose,color,depth,false);

        // Compute normals
        depth2normals(depth,normals,cam);

        Sample sample;
        sample.data = samplePatchWithScale(color,depth,normals,cam(0,2),cam(1,2),trans(2),cam(0,0),cam(1,1));

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

vector<Sample> datasetManager::createSynthSamplesWadim(Model &model,Matrix3f &cam, int index, int subdiv)
{
    // Create synthetic views
    SphereRenderer sphere(cam);
    Mat normals;
    Vector3f scales(0.4, 1.1, 1.0); // Render from 0.4 meters

    Vector3f in_plane_rots;
    if (inplane) { in_plane_rots = Vector3f(-45,15,45); }
            else { in_plane_rots = Vector3f(0,15,10); }

    vector<RenderView, Eigen::aligned_allocator<RenderView> > views =
            sphere.createViews(model,subdiv,scales,in_plane_rots,true,false);  // Equidistant sphere sampling with recursive level subdiv

    vector<Sample> samples;
    for (RenderView &v : views)
    {
        // Instead of taking object centroid, take the surface point as central sample point
        //        float z = v.dep.at<float>(cam(1,2),cam(0,2));
        //        assert(z>0.0f);
        float z = v.pose.translation()(2);

        depth2normals(v.dep,normals,cam);

        Sample sample;
        sample.data = samplePatchWithScale(v.col,v.dep,normals,cam(0,2),cam(1,2),z,cam(0,0),cam(1,1));

        // Build 6-dimensional label: model index + quaternion + translation
        sample.label = Mat(1,8,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(v.pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);
        for (size_t tr = 0; tr < 3; ++tr)
            sample.label.at<float>(0,5+tr) = v.pose.inverse().translation()(tr);

        samples.push_back(sample);
    }
    return samples;
}

void datasetManager::generateAndStoreSamples(int sampling_type)
{
    // Check if hdf5 files are available
    vector<string> missing_models;
    for (string &seq : used_models)
        if (!fexists(hdf5_path + "realSamples_" + seq + ".h5") ||
            !fexists(hdf5_path + "synthSamples_" + seq + ".h5") ||
            !fexists(hdf5_path + "templates_" + seq + ".h5"))
            missing_models.push_back(seq);

    for (string &model_name : missing_models) {

        clog << "\nCreating samples and patches for " << model_name << ":" << endl;

        // - load model
        Model model;
        model.loadPLY(dataset_path + model_name + ".ply");

        // - load frames of benchmark and visualize
        Benchmark bench;
        if (dataset_name == "LineMOD") {bench = loadLinemodBenchmark(dataset_path, model_name);}
        else if (dataset_name == "BigBIRD") {bench = loadBigbirdBenchmark(dataset_path, model_name);}
        else if (dataset_name == "Washington") {bench = loadLinemodBenchmark(dataset_path, model_name);}
        else {bench = loadLinemodBenchmark(dataset_path, model_name);}

        // Real data
        // - for each scene frame, extract RGBD sample
        vector<Sample> real_samples = extractRealSamplesPaul(bench.frames, bench.cam, model_index[model_name], model);

        // - store real samples to HDF5 files
        h5.write(hdf5_path + "realSamples_" + model_name +".h5", real_samples);
        // for (Sample &s : real_samples) showRGBDPatch(s.data);

        // Synthetic data
        clog << "  - render synthetic data:" << endl;
        vector<Sample> templates, synth_samples;
        if (sampling_type) {
            int subdivTmpl = 2; // sphere subdivision factor for templates
            templates = createSynthSamplesWadim(model, bench.cam, model_index[model_name], subdivTmpl);
            synth_samples = createSynthSamplesWadim(model, bench.cam, model_index[model_name], subdivTmpl+1);
        } else {
            vector<Sample> temp = createSynthSamplesPaul(model, bench.cam, model_index[model_name]);
            templates.assign(temp.begin(),temp.begin() + 301);
            synth_samples.assign(temp.begin() + 302, temp.end());
        }

        // - store synthetic samples to HDF5 files
        h5.write(hdf5_path + "templates_" + model_name + ".h5", templates);
        h5.write(hdf5_path + "synthSamples_" + model_name + ".h5", synth_samples);
//         for (Sample &s : templates) showRGBDPatch(s.data);
        // for (Sample &s : synth_samples) showRGBDPatch(s.data);
    }
}

void datasetManager::generateDatasets()
{
    // Generate the hdf5 files if missing
    generateAndStoreSamples(0);

    // Clear the sets
    training_set.clear();
    template_set.clear();
    test_set.clear();

    // Load backgrounds for further use
    if (random_background == 3)
        backgrounds = loadBackgrounds(bg_path);


    for (string &seq : used_models)
    {
        // Read the data from hdf5 files
        vector<Sample> train_real(h5.read(hdf5_path + "realSamples_" + seq + ".h5"));
        vector<Sample> train_synth(h5.read(hdf5_path + "synthSamples_" + seq + ".h5"));
        template_set.push_back(h5.read(hdf5_path + "templates_" + seq + ".h5"));
        test_set.push_back(vector<Sample>());

        // Compute sizes and quaternions
        unsigned int nr_template_poses = template_set[0].size();
        unsigned int nr_real_poses = train_real.size();

        // Find the closest templates for each real sample
        vector<vector<int>> maxSimTmpl(nr_template_poses, vector<int>());
        for (size_t real_sample = 0; real_sample < nr_real_poses; ++real_sample)
        {
            float best_dist = numeric_limits<float>::max();
            unsigned int sim_tmpl = 0;
            for (size_t tmpl = 0; tmpl < nr_template_poses; tmpl++)
            {
                float temp_dist = train_real[real_sample].getQuat().angularDistance(template_set.back()[tmpl].getQuat());
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

    // Compute MaxSimTmpls to build batches
    if (inplane) { computeMaxSimTmplInplane(); }
            else { computeMaxSimTmpl(); }
}

void datasetManager::computeMaxSimTmplInplane()
{
    size_t nr_training_poses = training_set.front().size();
    size_t nr_template_poses = template_set.front().size();

    // Calculate maxSimTmpl: find the 2 most similar templates for each object in the training set
    maxSimTmpl.assign(nr_objects, vector<vector<int>>(nr_training_poses, vector<int>()));
    for (size_t object = 0; object < nr_objects; ++object)
    {
        for (size_t training_pose = 0; training_pose < nr_training_poses; ++training_pose)
        {
            float best_dist = numeric_limits<float>::max();
            float best_dist2 = numeric_limits<float>::max(); // second best
            int sim_tmpl, sim_tmpl2;

            // - find the first most similar template
            for (size_t tmpl_pose = 0; tmpl_pose < nr_template_poses; tmpl_pose++)
            {
                float temp_dist = training_set[object][training_pose].getQuat().angularDistance(template_set[object][tmpl_pose].getQuat());
                if (temp_dist >= best_dist) continue;
                best_dist = temp_dist;
                sim_tmpl = tmpl_pose;
            }

            // - push back the template
            maxSimTmpl[object][training_pose].push_back(sim_tmpl);

            // - find the second most similar template
            for (size_t tmpl_pose = 0; tmpl_pose < nr_template_poses; tmpl_pose++)
            {
                float temp_dist = training_set[object][training_pose].getQuat().angularDistance(template_set[object][tmpl_pose].getQuat());
                if (temp_dist >= best_dist2 || temp_dist == best_dist) continue;
                best_dist2 = temp_dist;
                sim_tmpl2 = tmpl_pose;
            }

            // - push back the template
            maxSimTmpl[object][training_pose].push_back(sim_tmpl2);

#if 0
            imshow("query",showRGBDPatch(training_set[object][training_pose].data,false));
            imshow("sim 1",showRGBDPatch(template_set[object][maxSimTmpl[object][training_pose][0]].data,false));
            imshow("sim 2",showRGBDPatch(template_set[object][maxSimTmpl[object][training_pose][1]].data,false));
            waitKey();
#endif
        }
    }
}

void datasetManager::computeMaxSimTmpl()
{
    size_t nr_training_poses = training_set.front().size();
    size_t nr_template_poses = template_set.front().size();

    // Calculate maxSimTmpl: find the 2 most similar templates for each object in the training set
    maxSimTmpl.assign(nr_objects, vector<vector<int>>(nr_training_poses, vector<int>()));
    for (size_t object = 0; object < nr_objects; ++object)
    {
        for (size_t training_pose = 0; training_pose < nr_training_poses; ++training_pose)
        {
            float best_dist = numeric_limits<float>::min();
            float best_dist2 = numeric_limits<float>::min(); // second best
            int sim_tmpl, sim_tmpl2;

            // - find the first most similar template
            for (size_t tmpl_pose = 0; tmpl_pose < nr_template_poses; tmpl_pose++)
            {
                float temp_dist = training_set[object][training_pose].getTrans().dot(template_set[object][tmpl_pose].getTrans());
                if (temp_dist <= best_dist) continue;
                best_dist = temp_dist;
                sim_tmpl = tmpl_pose;
            }

            // - push back the template
            maxSimTmpl[object][training_pose].push_back(sim_tmpl);


            // - find the second most similar template
            for (size_t tmpl_pose = 0; tmpl_pose < nr_template_poses; tmpl_pose++)
            {
                float temp_dist = training_set[object][training_pose].getTrans().dot(template_set[object][tmpl_pose].getTrans());
                if (temp_dist <= best_dist2 || temp_dist == best_dist) continue;
                best_dist2 = temp_dist;
                sim_tmpl2 = tmpl_pose;
            }

            // - push back the template
            maxSimTmpl[object][training_pose].push_back(sim_tmpl2);

#if 0
            imshow("query",showRGBDPatch(training_set[object][training_pose].data,false));
            imshow("sim 1",showRGBDPatch(template_set[object][maxSimTmpl[object][training_pose][0]].data,false));
            imshow("sim 2",showRGBDPatch(template_set[object][maxSimTmpl[object][training_pose][1]].data,false));
            waitKey();
#endif
        }
    }
}

void datasetManager::randomFill(Mat &patch, int type)
{
    switch(type) {
        case 1: randomColorFill(patch); break;
        case 2: randomShapeFill(patch); break;
        case 3: randomRealFill(patch); break;
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
    // medianBlur(tmp_rgb, tmp_rgb, 5);
    GaussianBlur(tmp_rgb, tmp_rgb, Size(5,5), 0, 0);

    // Store the mask
    Mat mask = patch_dep == 0;

    // Copy random shapes to the background of the patch
    tmp_rgb.copyTo(patch_rgb,mask);
    tmp_dep.copyTo(patch_dep,mask);

    cv::merge(vector<Mat>{patch_rgb,patch_dep,patch_nor},patch);
    // showRGBDPatch(patch, true);

}

void datasetManager::randomRealFill(Mat &patch)
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

    // Store the mask
    Mat mask = patch_dep == 0;

    // Adjust depth
    float depth_scale = 0.45 / backgrounds[bg].depth.at<float>(center_x, center_y);
    tmp_dep *= depth_scale;

    // Fill backgrounds
    tmp_rgb.convertTo(tmp_rgb, CV_32FC3, 1/255.f);
    tmp_rgb.copyTo(patch_rgb,mask);
    tmp_dep.copyTo(patch_dep,mask);
    tmp_nor.copyTo(patch_nor,mask);

    medianBlur(patch_rgb, patch_rgb, 3);
    medianBlur(patch_nor, patch_nor, 3);

    cv::merge(vector<Mat>{patch_rgb,patch_dep,patch_nor},patch);
    // showRGBDPatch(patch, true);
}

datasetManager::datasetManager(string config)
{
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);
    dataset_path = pt.get<string>("paths.dataset_path");
    hdf5_path = pt.get<string>("paths.hdf5_path");
    bg_path = pt.get<string>("paths.background_path");

    dataset_name = pt.get<string>("input.dataset_name");
    random_background = pt.get<int>("input.random_background");
    use_real = pt.get<bool>("input.use_real");
    inplane = pt.get<bool>("input.inplane");
    models = to_array<string>(pt.get<string>("input.models"));
    used_models = to_array<string>(pt.get<string>("input.used_models"));
    rotInv = to_array<int>(pt.get<string>("input.rotInv"));
    nr_objects = used_models.size();

    // For each object build a mapping from model name to index number
    for (size_t i = 0; i < models.size(); ++i) model_index[models[i]] = i;
}
}
