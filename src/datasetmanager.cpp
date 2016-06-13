#include "datasetmanager.h"

datasetManager::datasetManager(string config)
{
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);
    dataset_path = pt.get<string>("paths.dataset_path");
    hdf5_path = pt.get<string>("paths.hdf5_path");

    models = to_array<string>(pt.get<string>("input.models"));
    used_models = to_array<string>(pt.get<string>("input.used_models"));
    rotInv = to_array<int>(pt.get<string>("input.rotInv"));
    nr_objects = used_models.size();

    // For each object build a mapping from model name to index number
    for (size_t i = 0; i < used_models.size(); ++i) model_index[used_models[i]] = i;
}

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
        if (filesystem::is_regular_file(dir_iter->status()) )
        {
            string file = dir_iter->path().leaf().string();
            if (file.substr(0,5)=="color")
                last = std::max(last,std::stoi(file.substr(5,file.length())));

        }

    if (count>-1) last = count;
//    cout << "  - loading frames in the range " << 0 << " - " << last << ":"<< endl;
    Benchmark bench;
    for (int i=0; i <= last;i++)
    {
        Frame frame;
        frame.nr = i;
        frame.color = imread(dir_string + "/color"+to_string(i)+".jpg");
        frame.depth = imread(dir_string + "/inp/depth"+to_string(i)+".png",-1);
        assert(!frame.color.empty() && !frame.depth.empty());
        frame.depth.convertTo(frame.depth,CV_32F,0.001f);   // Bring depth map into meters
        ifstream pose(dir_string + "/pose"+to_string(i)+".txt");
        assert(pose.is_open());
        for (int k=0; k < 4;k++)
            for(int l=0; l < 4;l++)
                pose >> frame.gt.matrix()(k,l);
        bench.frames.push_back(frame);
//        cout << sequence << ", sample:  " << i << endl;
        loadbar("  - loading frames: ",i,last);
    }

    bench.cam = Matrix3f::Identity();
    bench.cam(0,0) = 572.4114f;
    bench.cam(0,2) = 325.2611f;
    bench.cam(1,1) = 573.5704f;
    bench.cam(1,2) = 242.0489f;
    return bench;
}

// Takes a color and depth frame and samples a normalized 4-channel patch at the given center position and z-scale
Mat datasetManager::samplePatchWithScale(Mat &color, Mat &depth, int center_x, int center_y, float z, float fx, float fy)
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
    Mat temp_col, temp_dep, final;
    color(cut).copyTo(temp_col);
    depth(cut).copyTo(temp_dep);

    Mat bg = (temp_dep == 0);

    // Convert to float and rescale to [-1,1]
    temp_col.convertTo(temp_col,CV_32FC3,1/255.f);
    temp_col = (temp_col-0.5f)*2.f;

    // Demean with central z value, clamp and rescale to [-1,1]
    temp_dep -= z;
    temp_dep.setTo(-m, temp_dep < -m);
    temp_dep.setTo(m, temp_dep > m);
    temp_dep *= 1.0f / m;

    // Resize
    const int CNN_INPUT_SIZE = 64;
    const Size final_size(CNN_INPUT_SIZE,CNN_INPUT_SIZE);
    resize(temp_col,temp_col,final_size);   // Standard bilinear interpolation
    resize(temp_dep,temp_dep,final_size,0,0,INTER_NEAREST);// Nearest-neighbor interpolation for depth!!!

    cv::merge(vector<Mat>{temp_col,temp_dep},final);

    // Bring all back to [0,1]
    final = (final+1.f)*0.5f;

    return final;
}

vector<Sample> datasetManager::extractSceneSamplesPaul(vector<Frame,Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index)
{
    vector<Sample> samples;
    for (Frame &f : frames)
    {
        Vector3f centroid = f.gt.translation();
        Vector3f projCentroid = cam*centroid;
        projCentroid /= projCentroid(2);

        Sample s;
        s.data = samplePatchWithScale(f.color,f.depth,projCentroid(0),projCentroid(1),centroid(2),cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion
        s.label = Mat(1,5,CV_32F);
        s.label.at<float>(0,0) = index;
        Quaternionf q(f.gt.linear());
        for (int i=0; i < 4; ++i)
            s.label.at<float>(0,1+i) = q.coeffs()(i);

        samples.push_back(s);
    }
    return samples;
}

vector<Sample> datasetManager::extractSceneSamplesWadim(vector<Frame,Eigen::aligned_allocator<Frame>> &frames, Matrix3f &cam, int index)
{
    vector<Sample> samples;
    for (Frame &f : frames)
    {

        // Instead of taking object centroid, take the surface point as central sample point
        Vector3f projCentroid = cam*f.gt.translation();
        projCentroid /= projCentroid(2);

        float z = f.depth.at<float>(projCentroid(1),projCentroid(0));
        assert(z>0.0f);

        Sample s;
        s.data = samplePatchWithScale(f.color,f.depth,projCentroid(0),projCentroid(1),z,cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion
        s.label = Mat(1,5,CV_32F);
        s.label.at<float>(0,0) = index;
        Quaternionf q(f.gt.linear());
        for (int i=0; i < 4; ++i)
            s.label.at<float>(0,1+i) = q.coeffs()(i);

        samples.push_back(s);
    }
    return samples;
}

vector<Sample> datasetManager::createTemplatesPaul(Model &model, Matrix3f &cam, int index, int rotInv)
{

    ifstream file(dataset_path + "paul/camPositionsElAz.txt");
    assert(file.good());
    vector<Vector2f, Eigen::aligned_allocator<Vector2f> > sphereCoords(1542);

    // Read all camera poses from file (given in elevation and azimuth)
    vector<Matrix3f, Eigen::aligned_allocator<Matrix3f> > camPos;
    float last_el = 0;
    int coordId = 0;
    for(Vector2f &v : sphereCoords)
    {
        file >> v(0) >> v(1);
        if (rotInv == 1)
        {
            if (abs(last_el - v(0)) <= 0.01 && coordId != 0 && coordId < 302) {
                camPos.push_back(camPos.back());
            } else {
                // Copied from basetypes.py lines 100-xxx
                // To move into el=0,az=0 position we first have to rotate 45deg around x
                AngleAxisf camRot0(M_PI/2,Vector3f(1,0,0));

                // Build rotation matrix from spherical coordinates
                AngleAxisf el(v(0),Vector3f(1,0,0));
                AngleAxisf az(-v(1),Vector3f(0,0,1));
                Matrix3f camRot = (el*az).toRotationMatrix();
                camPos.push_back(camRot0*camRot);
            }
        } else if (rotInv == 2 && coordId < 302) {
            if (v(1) > 0) {
                // Copied from basetypes.py lines 100-xxx
                // To move into el=0,az=0 position we first have to rotate 45deg around x
                AngleAxisf camRot0(M_PI/2,Vector3f(1,0,0));

                // Build rotation matrix from spherical coordinates
                AngleAxisf el(v(0),Vector3f(1,0,0));
                AngleAxisf az(-v(1),Vector3f(0,0,1));
                Matrix3f camRot = (el*az).toRotationMatrix();
                camPos.push_back(camRot0*camRot);
            } else {
                camPos.push_back(camPos.back());
            }
        } else {
            // Copied from basetypes.py lines 100-xxx
            // To move into el=0,az=0 position we first have to rotate 45deg around x
            AngleAxisf camRot0(M_PI/2,Vector3f(1,0,0));

            // Build rotation matrix from spherical coordinates
            AngleAxisf el(v(0),Vector3f(1,0,0));
            AngleAxisf az(-v(1),Vector3f(0,0,1));
            Matrix3f camRot = (el*az).toRotationMatrix();
            camPos.push_back(camRot0*camRot);
        }
        last_el = v(0);
        coordId++;
    }

    // Render each and create proper sample
    SphereRenderer renderer;
    renderer.init(cam);
    Mat col,dep;
    Isometry3f pose = Isometry3f::Identity();
    pose.translation()(2) = 0.4f;

    vector<Sample> samples;
    for (Matrix3f &m : camPos)
    {
        pose.linear() = m;
        renderer.renderView(model,pose,col,dep,false);

        Sample sample;
        sample.data = samplePatchWithScale(col,dep,cam(0,2),cam(1,2),pose.translation()(2),cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion
        sample.label = Mat(1,5,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);

        samples.push_back(sample);
    }
    return samples;
}

vector<Sample> datasetManager::createTemplatesWadim(Model &model,Matrix3f &cam, int index, int rotInv, int subdiv)
{

    // Create synthetic views
    SphereRenderer sphere(cam);
    Vector3f scales(0.4, 1.1, 1.0);     // Render from 0.4 meters
    Vector3f in_plane_rots(0,15,10);  // Render in_plane_rotations from -45 degree to 45 degree in 15degree steps
    vector<RenderView, Eigen::aligned_allocator<RenderView> > views =
            sphere.createViews(model,subdiv,scales,in_plane_rots,true,false,rotInv, subdiv);    // Equidistant sphere sampling with recursive level subdiv

    vector<Sample> samples;
    for (RenderView &v : views)
    {
        // Instead of taking object centroid, take the surface point as central sample point
        //        float z = v.dep.at<float>(cam(1,2),cam(0,2));
        //        assert(z>0.0f);

        Sample sample;
        sample.data = samplePatchWithScale(v.col,v.dep,cam(0,2),cam(1,2),v.pose.translation()(2),cam(0,0),cam(1,1));

        // Build 5-dimensional label: model index + quaternion
        sample.label = Mat(1,5,CV_32F);
        sample.label.at<float>(0,0) = index;
        Quaternionf q(v.pose.linear());
        for (int i=0; i < 4; ++i)
            sample.label.at<float>(0,1+i) = q.coeffs()(i);

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
        Benchmark bench = loadLinemodBenchmark(dataset_path, model_name);
        // showGT(bench,model);

        // Real data
        // - for each scene frame, extract RGBD sample
        vector<Sample> realSamples = extractSceneSamplesPaul(bench.frames,bench.cam,model_index[model_name]);

        // - store realSamples to HDF5 files
        h5.write(hdf5_path + "realSamples_" + model_name +".h5", realSamples);
        // for (Sample &s : realSamples) showRGBDPatch(s.data);

        // Synthetic data
        clog << "  - render synthetic data:" << endl;
        // - create synthetic samples and templates
        int subdivTmpl = 3; // sphere subdivision factor for templates
        vector<Sample> templates = createTemplatesWadim(model, bench.cam, model_index[model_name], rotInv[model_index[model_name]], subdivTmpl);
        vector<Sample> synthSamples = createTemplatesWadim(model, bench.cam, model_index[model_name], 0, subdivTmpl+1);
        //  vector<Sample> temp = createTemplatesPaul(model, bench.cam, model_index[model_name], rotInv[model_index[model_name]]);
        //  vector<Sample> templates (temp.begin(),temp.begin() + 301);
        //  vector<Sample> synthSamples (temp.begin() + 302, temp.end());

        // - store realSamples to HDF5 files
        h5.write(hdf5_path + "templates_" + model_name + ".h5", templates);
        h5.write(hdf5_path + "synthSamples_" + model_name + ".h5", synthSamples);
        // for (Sample &s : templates) showRGBDPatch(s.data);

    }
}

void datasetManager::generateDatasets()
{
    training_set.clear();
    test_set.clear();
    templates.clear();


    for (string &seq : used_models)
    {
        // Read the data from hdf5 files
        vector<Sample> train_real(h5.read(hdf5_path + "realSamples_" + seq + ".h5"));
        vector<Sample> train_synth(h5.read(hdf5_path + "synthSamples_" + seq + ".h5"));
        templates.push_back(h5.read(hdf5_path + "templates_" + seq + ".h5"));
        test_set.push_back(vector<Sample>());

        // Compute sizes and quaternions
        unsigned int nr_template_poses = templates[0].size();
        unsigned int nr_synth_poses = train_synth.size();
        unsigned int nr_real_poses = train_real.size();

        // - read quaternion poses from templates (they are identical for all objects)
        vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > tmpl_quats(nr_template_poses);
        for  (size_t i=0; i < nr_template_poses; ++i)
            for (size_t j=0; j < 4; ++j)
                tmpl_quats[i].coeffs()(j) = templates[0][i].label.at<float>(0,1+j);

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
                for (size_t i = 0; i < ceil(maxSimTmpl[tmpl].size()/2.0); ++i) {
                    train_synth.push_back(train_real[maxSimTmpl[tmpl][i]]);
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
    for (int object = 0; object < nr_objects; ++object) {
        if (min_training >= training_set[object].size()) min_training = training_set[object].size();
        if (min_test >= test_set[object].size()) min_test = test_set[object].size();
    }
    for (int object = 0; object < nr_objects; ++object) {
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
    tmpl_quats.assign(nr_template_poses, Quaternionf());
    for  (size_t i = 0; i < nr_template_poses; ++i)
        for (size_t j = 0; j < 4; ++j)
            tmpl_quats[i].coeffs()(j) = templates[0][i].label.at<float>(0,1+j);

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

void datasetManager::addNoiseToSynthData(unsigned int copies, vector<vector<Sample>>& trainingSet)
{
    for (size_t copy = 0; copy < copies; ++copy) {
        for (size_t object = 0; object < trainingSet.size(); ++object) {
            for (size_t pose = 0; pose < trainingSet[object].size(); ++pose) {
                // create a new image, add noise, push it back to the training set
//                trainingSet[object][pose].push_back(noise_image);
            }
        }
    }

}

