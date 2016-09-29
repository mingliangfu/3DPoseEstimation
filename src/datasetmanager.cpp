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

    Benchmark bench;

    // Read intristic cameras
    bench.cam = h5.readBBIntristicMats(dir_string + "/calibration.h5");

    // Read transformations
    vector<Isometry3f,Eigen::aligned_allocator<Isometry3f>> trans = h5.readBBTrans(dir_string + "/calibration.h5");

    for (int np = 1 ; np <= 5; ++np) {
        for (int i = 0; i <= 357; i += 3)
        {
            Frame frame;
            frame.nr = i * np;
            frame.color = imread(dir_string + "/NP" + to_string(np) + "_" + to_string(i)+".jpg");
            // resize(frame.color, frame.color, frame.color.size());
            // imshow("ColorScaled: ", frame.color); waitKey();

            frame.depth = imread(dir_string + "/NP" + to_string(np) + "_" + to_string(i)+".png", -1);
            frame.depth.convertTo(frame.depth, CV_32F, 0.0001f);

            // Filter depth
            Mat depth_mini(frame.depth.size().height, frame.depth.size().width, CV_8UC1);
            frame.depth.convertTo(depth_mini, CV_8UC1, 255.0);
            resize(depth_mini, depth_mini, Size(), 0.2, 0.2);
            cv::inpaint(depth_mini, (depth_mini == 0.0), depth_mini, 5.0, INPAINT_TELEA);
            resize(depth_mini, depth_mini, frame.depth.size());
            depth_mini.convertTo(depth_mini, CV_32FC1, 1./255.0);
            depth_mini.copyTo(frame.depth, (frame.depth == 0));

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

Benchmark datasetManager::loadWashingtonBenchmark(string washington_path, string sequence, int count /*=-1*/)
{
    string dir_string = washington_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;
    Benchmark bench;

    ifstream pose(dir_string + "/poses.txt");
    assert(pose.is_open());

    int num;
    while(pose >> num)
    {
        Frame frame;
        frame.nr = num;
        stringstream filenum;
        filenum << setw(6) << setfill('0') << num;
        frame.color = imread(dir_string + "/color_" + filenum.str() + ".png");
        frame.depth = imread(dir_string + "/depth_" + filenum.str() + ".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);   // Bring depth map into meters
        frame.gt.push_back({sequence,Isometry3f::Identity()});

        imshow("Color: ", frame.color); waitKey();

        // Read pose
        for (int k=0; k < 4;k++)
            for(int l=0; l < 4;l++)
                pose >> frame.gt[0].second.matrix()(k,l);
        cout << frame.gt[0].second.matrix() << endl;
        bench.frames.push_back(frame);
    }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;
    return bench;
}

Benchmark datasetManager::loadBenjaminBenchmark(string benjamin_path, string sequence)
{
    string dir_string = benjamin_path + sequence;
    cerr << "  - loading benchmark " << dir_string << endl;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return Benchmark();
    }

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
                    frame.depth.at<float>(temprow, tempcol) = (float)m*0.001f;
                }

                // Add pose matrix
                ifstream pose(dir_string + "/" + file.substr(0, file.find("_color.png")) + "_pose.txt");
                assert(pose.is_open());

                frame.gt.push_back({sequence,Isometry3f::Identity()});

                for (int k=0; k < 4;k++)
                    for(int l=0; l < 4;l++)
                        pose >> frame.gt[0].second.matrix()(k,l);

                Vector3f trans = Vector3f(0,0,1);
                frame.gt[0].second.translation() = trans;

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

vector<Sample> datasetManager::extractSynthSamplesBenjamin(string benjamin_path, Matrix3f &cam, string sequence, int index)
{
    string dir_string = benjamin_path + sequence;

    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return vector<Sample>();
    }

    vector<Sample> samples;
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
                    frame.depth.at<float>(temprow, tempcol) = (float)m*0.001f;
                }

                // Add pose matrix
                ifstream pose(dir_string + "/" + file.substr(0, file.find("_color.png")) + "_pose.txt");
                assert(pose.is_open());

                frame.gt.push_back({sequence,Isometry3f::Identity()});

                for (int k=0; k < 4;k++)
                    for(int l=0; l < 4;l++)
                        pose >> frame.gt[0].second.matrix()(k,l);

                Vector3f trans = Vector3f(0,0,1);
                frame.gt[0].second.translation() = trans;

                // Generate sample
                Vector3f centroid = frame.gt[0].second.translation();
                Vector3f projCentroid = cam * centroid;
                projCentroid /= projCentroid(2);

                // Compute normals
                depth2normals(frame.depth,frame.normals,cam);

                Sample sample;
                sample.data = samplePatchWithScale(frame.color,frame.depth,frame.normals,projCentroid(0),projCentroid(1),centroid(2),cam(0,0),cam(1,1));

                // Build 5-dimensional label: model index + quaternion + translation
                sample.label = Mat(1,8,CV_32F);
                sample.label.at<float>(0,0) = index;
                Quaternionf q(frame.gt[0].second.linear());
                for (int i=0; i < 4; ++i)
                    sample.label.at<float>(0,1+i) = q.coeffs()(i);
                for (size_t tr = 0; tr < 3; ++tr)
                    sample.label.at<float>(0,5+tr) = frame.gt[0].second.inverse().translation()(tr);

                samples.push_back(sample);
            }
        }

    return samples;
}

// Takes a color and depth frame and samples a normalized 4-channel patch at the given center position and z-scale
Mat datasetManager::samplePatchWithScale(Mat &color, Mat &depth, Mat &normals, int center_x, int center_y, float z, float fx, float fy)
{
    // Make a cut of metric size m
    float m;
    if (dataset_name == "LineMOD") {m = 0.2f;}
    else if (dataset_name == "BigBIRD") {m = 0.2f;}
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

void datasetManager::loadLinemodHardNegatives()
{
    for (string &s : used_models)
    {
        string file = hdf5_path + "negs_" + s + ".h5";
        if (!boost::filesystem::exists(file))
            //throw runtime_error(file + " not found!");
            cerr << file + " not found!" << endl;
        else hard_negatives[s] = h5.read(file);
    }
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
        // imshow("Model", color); waitKey();

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
        Model model; int type; string ext;

        if (fexists(dataset_path + model_name + ".ply")) {type = 1; ext = ".ply";}
        else if (fexists(dataset_path + model_name + ".obj")) {type = 2; ext = ".obj";}
        else {throw runtime_error("No model found for " + model_name + "!");}
        model.loadModel(dataset_path + model_name + ext, type);

        // - load frames of benchmark and visualize
        Benchmark bench;
        if (dataset_name == "LineMOD") {bench = loadLinemodBenchmark(dataset_path, model_name);}
        else if (dataset_name == "BigBIRD") {bench = loadBigbirdBenchmark(dataset_path, model_name);}
        else if (dataset_name == "Washington") {bench = loadWashingtonBenchmark(dataset_path, model_name);}
        else {bench = loadLinemodBenchmark(dataset_path, model_name);}

        // Real data
        // - for each scene frame, extract RGBD sample
        vector<Sample> real_samples = extractRealSamplesPaul(bench.frames, bench.cam, model_index[model_name], model);

        // - store realSamples to HDF5 files
        h5.write(hdf5_path + "realSamples_" + model_name +".h5", real_samples);
        //for (Sample &s : realSamples) showRGBDPatch(s.data,true);

        // Synthetic data
        clog << "  - render synthetic data:" << endl;
        vector<Sample> templates, synth_samples;
        if (sampling_type == 0) {
            vector<Sample> temp = createSynthSamplesPaul(model, bench.cam, model_index[model_name]);
            templates.assign(temp.begin(),temp.begin() + 301);
            synth_samples.assign(temp.begin() + 302, temp.end());
        } else if (sampling_type == 1) {
            int subdivTmpl = 2; // sphere subdivision factor for templates
            templates = createSynthSamplesWadim(model, bench.cam, model_index[model_name], subdivTmpl);
            synth_samples = createSynthSamplesWadim(model, bench.cam, model_index[model_name], subdivTmpl+1);
        } else if (sampling_type == 2) {
            string simulated_templates_path = "/media/zsn/My Passport/Datasets/Benjamin/dataset_sergey_poses/";
            string simulated_training_set_path = "/media/zsn/My Passport/Datasets/Benjamin/dataset_sergey_poses/";
            templates = extractSynthSamplesBenjamin(simulated_templates_path, bench.cam, model_name, model_index[model_name]);
            synth_samples = extractSynthSamplesBenjamin(simulated_training_set_path, bench.cam, model_name, model_index[model_name]);
        }

        // - store synthetic samples to HDF5 files
        h5.write(hdf5_path + "templates_" + model_name + ".h5", templates);
        h5.write(hdf5_path + "synthSamples_" + model_name + ".h5", synth_samples);
        //for (Sample &s : templates) showRGBDPatch(s.data,true);
        //for (Sample &s : synth_samples) showRGBDPatch(s.data,true);

    }
}

void datasetManager::generateDatasets()
{
    // Generate the hdf5 files if missing
    if (use_simulated) generateAndStoreSamples(2);
    else generateAndStoreSamples(inplane);

    // Clear the sets
    training_set.clear();
    template_set.clear();
    test_set.clear();

    // Load backgrounds for further use
    if (random_background == 3 || random_background == -1)
        bg.loadBackgrounds(bg_path);

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

    // For LineMOD, load the hard negatives
    if (dataset_name == "LineMOD")
            loadLinemodHardNegatives();

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
    if (type==-1)
    {
        std::uniform_int_distribution<int> lol(1,3);
        type = lol(ran);
    }

    switch(type) {
        case 1: bg.randomColorFill(patch); break;
        case 2: bg.randomShapeFill(patch); break;
        case 3: bg.randomFractalFill(patch); break;
        case 4: bg.randomRealFill(patch); break;
    }
}

datasetManager::datasetManager(string config)
{
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);

    dataset_path = pt.get<string>("paths.dataset_path");
    hdf5_path = pt.get<string>("paths.hdf5_path");
    bg_path = pt.get<string>("paths.background_path");

    use_simulated = pt.get<bool>("input.use_simulated");
    simulated_templates_path = pt.get<string>("paths.simulated_templates_path");
    simulated_templates_path = pt.get<string>("paths.simulated_training_path");

    dataset_name = pt.get<string>("input.dataset_name");
    random_background = pt.get<int>("input.random_background");
    use_real = pt.get<bool>("input.use_real");
    inplane = pt.get<bool>("input.inplane");
    models = to_array<string>(pt.get<string>("input.models"));
    used_models = to_array<string>(pt.get<string>("input.used_models"));
    rotInv = to_array<int>(pt.get<string>("input.rotInv"));
    nr_objects = used_models.size();

    if ((dataset_name != "LineMOD") &&
        (dataset_name != "BigBIRD") &&
        (dataset_name != "Washington"))
        throw runtime_error("Unknown dataset: " + dataset_name + "!");

    // For each object build a mapping from model name to index number
    for (size_t i = 0; i < models.size(); ++i) model_index[models[i]] = i;
}
}
