#include "networksolver.h"

networkSolver::networkSolver(string config)
{
    // Read learning parameters
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);

    network_path = pt.get<string>("paths.network_path");
    net_name = pt.get<string>("train.net_name");
    num_epochs = pt.get<unsigned int>("train.num_epochs");
    num_training_rounds = pt.get<unsigned int>("train.num_training_rounds");
    learning_rate =  pt.get<float>("train.learning_rate");
    momentum = pt.get<float>("train.momentum");
    weight_decay =  pt.get<float>("train.weight_decay");
    learning_policy = pt.get<string>("train.learning_policy");
    step_size = pt.get<unsigned int>("train.step_size");
    gamma = pt.get<float>("train.gamma");
    gpu = pt.get<bool>("train.gpu");

    if (gpu) caffe::Caffe::set_mode(caffe::Caffe::GPU);

    used_models = to_array<string>(pt.get<string>("input.used_models"));

    // Generate the datasets out of the stored h5 files
    db_manager = new datasetManager(config);

}

void networkSolver::generateDataset()
{
    db_manager->generateDatasets(training_set, test_set, templates);

    // Save dataset parameters
    nr_objects = used_models.size();
    nr_training_poses = training_set[0].size();
    nr_template_poses = templates[0].size();
    nr_test_poses = test_set[0].size();

    // For each object build a mapping from model name to index number
    for (size_t i = 0; i < nr_objects; ++i) model_index[used_models[i]] = i;

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

    // Calculate maxSimTmpl: find the 2 most similar templates for each object in the training set
    maxSimTmpl.assign(nr_objects, vector<vector<int>>(nr_training_poses, vector<int>()));
    for (int object = 0; object < nr_objects; ++object)
    {
        for (int training_pose = 0; training_pose < nr_training_poses; ++training_pose)
        {
            float best_dist = numeric_limits<float>::max();
            float best_dist2 = numeric_limits<float>::max(); // second best
            int sim_tmpl;

            // - push back the first most similar template
            for (size_t tmpl_pose = 0; tmpl_pose < nr_template_poses; tmpl_pose++)
            {
                float temp_dist = training_quats[object][training_pose].angularDistance(tmpl_quats[tmpl_pose]);
                if (temp_dist >= best_dist) continue;
                best_dist = temp_dist;
                sim_tmpl = tmpl_pose;
            }
            maxSimTmpl[object][training_pose].push_back(sim_tmpl);

            // - push back the second most similar template
            for (size_t tmpl_pose1 = 0; tmpl_pose1 < nr_template_poses; tmpl_pose1++)
            {
                float temp_dist = training_quats[object][training_pose].angularDistance(tmpl_quats[tmpl_pose1]);
                if (temp_dist >= best_dist2 || temp_dist == best_dist) continue;
                best_dist2 = temp_dist;
                sim_tmpl = tmpl_pose1;
            }
            maxSimTmpl[object][training_pose].push_back(sim_tmpl);
        }
    }
}

vector<Sample> networkSolver::buildBatch(int batch_size, int iter, bool bootstrapping)
{
    int triplet_size = 5;
    vector<Sample> batch;
    size_t puller = 0, pusher0 = 0, pusher1 = 0, pusher2 = 0;
    TripletWang triplet;

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_template_poses-1);

    for (int linearId = iter * batch_size/triplet_size; linearId < (iter * batch_size/triplet_size) + batch_size/triplet_size; ++linearId) {

        // Calculate 2d indices
        unsigned int training_pose = linearId / nr_objects;
        unsigned int object = linearId % nr_objects;

        // Anchor: training set sample
        triplet.anchor = training_set[object][training_pose];
        // Puller: most similar template
        puller = maxSimTmpl[object][training_pose][0];
        triplet.puller = templates[object][puller];
        // Pusher 0: second most similar template
        pusher0 = maxSimTmpl[object][training_pose][1];
        triplet.pusher0 = templates[object][pusher0];

        if (bootstrapping)
        {
            // Pusher 1: fill either with the missclassified nn or random template of the same class
            unsigned int knn_object = maxSimKNNTmpl[object][training_pose][0];
            unsigned int knn_pose = maxSimKNNTmpl[object][training_pose][1];

            if (knn_object != object || knn_pose != puller) {
                // - missclassified nearest neighbor
                triplet.pusher1 = templates[knn_object][knn_pose];
            } else {
                // - random template of the same class
                pusher1 = ran_tpl(ran);
                while (pusher1 == puller && pusher1 == pusher0) pusher1 = ran_tpl(ran);
                triplet.pusher1 = templates[object][pusher1];
            }

            // Pusher 2: fill either with the missclassified knn or random template of a different class
            if (maxSimKNNTmpl[object][training_pose].size() > 2 && (knn_object != object || knn_pose != puller)) {
                // - missclassified knn of a different object
                knn_object = maxSimKNNTmpl[object][training_pose][2];
                knn_pose = maxSimKNNTmpl[object][training_pose][3];
                triplet.pusher2 = templates[knn_object][knn_pose];
            } else {
                // - template from another object
                pusher2 = ran_obj(ran);
                while (pusher2 == object) pusher2 = ran_obj(ran);
                triplet.pusher2 = templates[pusher2][ran_tpl(ran)];
            }
        } else {

            // Find pusher1: random template
            pusher1 = ran_tpl(ran);
            while (pusher1 == puller && pusher1 == pusher0) pusher1 = ran_tpl(ran);
            triplet.pusher1 = templates[object][pusher1];

            // Find pusher2: template from another object
            pusher2 = ran_obj(ran);
            while (pusher2 == object) pusher2 = ran_obj(ran);
            triplet.pusher2 = templates[pusher2][ran_tpl(ran)];
        }

#if 0   // Show triplets
        imshow("anchor",showRGBDPatch(triplet.anchor.data,false));
        imshow("puller",showRGBDPatch(triplet.puller.data,false));
        imshow("pusher0",showRGBDPatch(triplet.pusher0.data,false));
        imshow("pusher1",showRGBDPatch(triplet.pusher1.data,false));
        imshow("pusher2",showRGBDPatch(triplet.pusher2.data,false));
        if (bootstrapping) waitKey();
#endif

        // Store triplet to the batch
        batch.push_back(triplet.anchor);
        batch.push_back(triplet.puller);
        batch.push_back(triplet.pusher0);
        batch.push_back(triplet.pusher1);
        batch.push_back(triplet.pusher2);
    }
    return batch;
}

void networkSolver::trainNet(int resume_iter)
{

    // Set network parameters
    caffe::SolverParameter solver_param;
    solver_param.set_base_lr(learning_rate);
    solver_param.set_momentum(momentum);
    solver_param.set_weight_decay(weight_decay);
    solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);
    solver_param.set_lr_policy(learning_policy);
    solver_param.set_stepsize(step_size);
    solver_param.set_gamma(gamma);
    solver_param.set_snapshot_prefix(net_name);
    solver_param.set_display(1);
    solver_param.set_net(network_path + net_name + ".prototxt");
    caffe::SGDSolver<float> solver(solver_param);

    if (resume_iter > 0) {
        string resume_file = network_path + net_name + "_iter_" + to_string(resume_iter) + ".solverstate";
        solver.Restore(resume_file.c_str());
    }

    // Store the test network
    caffe::Net<float> testCNN(network_path + net_name + ".prototxt", caffe::TEST);

    // Get network information
    boost::shared_ptr<caffe::Net<float> > net = solver.net();
    caffe::Blob<float>* input_data_layer = net->input_blobs()[0];
    const size_t batch_size = input_data_layer->num();
    const int channels =  input_data_layer->channels();
    const int targetSize = input_data_layer->height();
    const int slice = input_data_layer->height()*input_data_layer->width();
    const int img_size = slice*channels;
    vector<float> data(batch_size*img_size,0), labels(batch_size,0);

    vector<Sample> batch;
    int triplet_size = 5;
    int epoch_iter = nr_objects * nr_training_poses / (batch_size/triplet_size);
    bool bootstrapping = false;

    // Perform training
    for (int training_round = 0; training_round < num_training_rounds; ++training_round)
    {
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            for (int iter = 0; iter < epoch_iter; iter++)
            {
                // Fill current batch
                batch = buildBatch(batch_size, iter, bootstrapping);

                // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
                for (size_t i=0; i < batch.size(); ++i)
                {
                    int currImg = i*img_size;
                    for (int ch = 0; ch < channels ; ++ch)
                        for (int y = 0; y < targetSize; ++y)
                            for (int x = 0; x < targetSize; ++x) {
                                data[currImg + slice*ch + y*targetSize + x] = batch[i].data.ptr<float>(y)[x*channels + ch];
                                labels[i] = batch[i].label.at<float>(0,0); }
                }
                input_data_layer->set_cpu_data(data.data());
                solver.Step(1);
            }
        }

        // Do bootstraping
        solver.Snapshot();
        int snapshot_iter = epoch_iter * num_epochs * (training_round + 1);
        testCNN.CopyTrainedLayersFrom(net_name + "_iter_" + to_string(snapshot_iter) + ".caffemodel");
        bootstrapping = bootstrap(testCNN, snapshot_iter);
    }
    clog << "Training finished!" << endl;
}

bool networkSolver::bootstrap(caffe::Net<float> &CNN, int iter)
{
    clog << "Bootstrapping: " << endl;
    clog << " - save the manifold png" << endl;
    visualizeManifold(CNN, iter);
    clog << " - save k-NN for all the training samples" << endl;
    computeKNN(CNN);
    return true;
}

Mat networkSolver::computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples)
{
    caffe::Blob<float>* input_layer = CNN.input_blobs()[0];
    const size_t batch_size = input_layer->num();
    const int channels =  input_layer->channels();
    const int targetSize = input_layer->height();
    const int slice = input_layer->height()*input_layer->width();
    const int img_size = slice*channels;

    caffe::Blob<float>* output_layer = CNN.output_blobs()[0];
    const size_t desc_dim = output_layer->channels();

    vector<float> data(batch_size*img_size,0);
    vector<int> currIdxs;
    Mat descs(samples.size(),desc_dim,CV_32F);

    currIdxs.reserve(batch_size);
    for (size_t i=0; i < samples.size(); ++i)
    {
        // Collect indices of samples to be processed for this batch
        currIdxs.push_back(i);
        if (currIdxs.size() == batch_size || i == samples.size()-1) // If batch full or last sample
        {
            // Fill linear batch memory with input data in Caffe layout with channel-first
            for (size_t j=0; j < currIdxs.size(); ++j)
            {
                Mat &patch = samples[currIdxs[j]].data;
                int currImg = j*img_size;
                for (int ch=0; ch < channels ; ++ch)
                    for (int y = 0; y < targetSize; ++y)
                        for (int x = 0; x < targetSize; ++x)
                            data[currImg + slice*ch + y*targetSize + x] = patch.ptr<float>(y)[x*channels + ch];
            }
            // Copy data memory into Caffe input layer, process batch and copy result back
            input_layer->set_cpu_data(data.data());
            vector< caffe::Blob<float>* > out = CNN.ForwardPrefilled();

            for (size_t j=0; j < currIdxs.size(); ++j)
                memcpy(descs.ptr<float>(currIdxs[j]), out[0]->cpu_data() + j*desc_dim, desc_dim*sizeof(float));

            currIdxs.clear(); // Throw away current batch
        }
    }
    return descs;
}


void networkSolver::visualizeManifold(caffe::Net<float> &CNN, int iter)
{
    Mat DBfeats;
    for (size_t tmpl = 0; tmpl < templates.size(); ++tmpl) {
        DBfeats.push_back(computeDescriptors(CNN, templates[tmpl]));
    }
    int nr_templates = DBfeats.rows/used_models.size();

    // If more than 3 dims, do PCA
    if (DBfeats.cols > 3)
    {

        PCA pca(DBfeats, Mat(), CV_PCA_DATA_AS_ROW);
        Mat proj = pca.project(DBfeats);
        proj(Rect(0,0,3,proj.rows)).copyTo(DBfeats);
    }

    // Visualize for the case where feat_dim is 3D
    viz::Viz3d visualizer("Manifold");
    cv::Mat vizMat(DBfeats.rows,1,CV_32FC3, DBfeats.data);
    cv::Mat colorMat;
    std::uniform_int_distribution<size_t> color(0, 255);
    for (size_t model = 0; model < used_models.size(); ++model) {
        colorMat.push_back(Mat(nr_templates,1,CV_8UC3,Scalar(color(ran), color(ran), color(ran))));
    }

    viz::WCloud manifold(vizMat,colorMat);
    manifold.setRenderingProperty(viz::POINT_SIZE, 5);
    visualizer.setBackgroundColor(cv::viz::Color::white(), cv::viz::Color::white());
    visualizer.setWindowSize(cv::Size(600, 400));
    visualizer.showWidget("cloud", manifold);
    //visualizer.spin();
    visualizer.spinOnce();
    visualizer.saveScreenshot("manifold_" + to_string(iter) + ".png");
}

void networkSolver::visualizeKNN(caffe::Net<float> &CNN, vector<string> test_models)
{
    // Get the test data: subset of the used_models
    Mat DBfeats, DBtest;
    vector<int> global_object_id;
    for (string &seq : test_models)
    {
        DBtest.push_back(computeDescriptors(CNN, test_set[model_index[seq]]));
        DBfeats.push_back(computeDescriptors(CNN, templates[model_index[seq]]));
        global_object_id.push_back(model_index[seq]);
    }

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    matcher->add(DBfeats);

    // Enter the infinite loop
    while(true)
    {
        // - get the random index for the query object
        uniform_int_distribution<size_t> DBtestGen(0, DBtest.rows - 1);
        int DBtestId = DBtestGen(ran);
        cout << DBtestId << endl;
        Mat testDescr = DBtest.row(DBtestId).clone();

        // - match the DBtest
        vector< vector<DMatch> > matches;
        int knn = 5;
        matcher->knnMatch(testDescr, matches, knn);

        // - get the test set indices (1D -> 2D)
        int query_object = global_object_id[DBtestId / test_set[0].size()];
        int query_sample = DBtestId % test_set[0].size();

        // - show the query object
        imshow("query",showRGBDPatch(test_set[query_object][query_sample].data,false));

        for (DMatch &m : matches[0])
        {
            // - get the template set indices (1D -> 2D)
            int tmpl_object = global_object_id[m.trainIdx / templates[0].size()];
            int tmpl_sample = m.trainIdx % templates[0].size();

            // - show the k-NN object
            imshow("kNN",showRGBDPatch(templates[tmpl_object][tmpl_sample].data,false));

            // - get the quaternions of the compared objects
            Quaternionf queryQuat, kNNQuat;
            for (int i=0; i < 4; ++i)
            {
                queryQuat.coeffs()(i) = test_set[query_object][query_sample].label.at<float>(0,1+i);
                kNNQuat.coeffs()(i) = templates[tmpl_object][tmpl_sample].label.at<float>(0,1+i);
            }

            // - compute angular difference
            bool object_match = (test_set[query_object][query_sample].label.at<float>(0,0) == templates[tmpl_object][tmpl_sample].label.at<float>(0,0));
            cout << "Object match: " << object_match << "; Angular difference: " << queryQuat.angularDistance(kNNQuat)*180.f/M_PI << endl;
            waitKey();
        }
    }
}

void networkSolver::computeKNN(caffe::Net<float> &CNN)
{
    // Get the training data
    Mat DBfeats, DBtraining;
    for (string &seq : used_models)
    {
        DBtraining.push_back(computeDescriptors(CNN, training_set[model_index[seq]]));
        DBfeats.push_back(computeDescriptors(CNN, templates[model_index[seq]]));
    }

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    matcher->add(DBfeats);

    // - match the DBtrainng
    vector< vector<DMatch> > matches;
    int knn = 3;
    matcher->knnMatch(DBtraining, matches, knn);

    maxSimKNNTmpl.assign(nr_objects, vector<vector<int>>(nr_training_poses, vector<int>()));

    for (size_t linearId = 0; linearId < DBtraining.rows; ++linearId) {

        // - get the test set indices (1D -> 2D)
        int query_object = linearId / nr_training_poses;
        int query_pose = linearId % nr_training_poses;

        for (int nn = 0; nn < knn; nn++)
        {
            int tmpl_object =  matches[linearId][nn].trainIdx / templates[0].size();
            int tmpl_pose =  matches[linearId][nn].trainIdx % templates[0].size();
#if 0
            imshow("query",showRGBDPatch(training_set[query_object][query_pose].data,false));
            imshow("simKNN",showRGBDPatch(templates[tmpl_object][tmpl_pose].data,false));
            waitKey();
#endif

            if (nn == 0) {
                // - store the object and the pose to the maxSimKNNTmpl
                maxSimKNNTmpl[query_object][query_pose].push_back(tmpl_object);
                maxSimKNNTmpl[query_object][query_pose].push_back(tmpl_pose);
            } else {
                if (maxSimKNNTmpl[query_object][query_pose].size() < 4 && tmpl_object != query_object) {
                    // - store the object and the pose to the maxSimKNNTmpl
                    maxSimKNNTmpl[query_object][query_pose].push_back(tmpl_object);
                    maxSimKNNTmpl[query_object][query_pose].push_back(tmpl_pose);
                }
            }
        }
    }
}

void networkSolver::evaluateNetwork(caffe::Net<float> &CNN)
{
    computeKNN(CNN);

    int intra = 0, inter = 0;
    for (int object = 0; object < nr_objects; ++object)
    {
        for (int training_pose = 0; training_pose < nr_training_poses; ++training_pose)
        {
#if 0
            cout << maxSimTmpl[object][training_pose][0] << " == " << maxSimKNNTmpl[object][training_pose][1]<< endl;
            imshow("query",showRGBDPatch(training_set[object][training_pose].data,false));
            imshow("sim",showRGBDPatch(templates[object][maxSimTmpl[object][training_pose][0]].data,false));
            imshow("simKnn",showRGBDPatch(templates[object][maxSimKNNTmpl[object][training_pose][1]].data,false));
            waitKey();
#endif
            if (maxSimTmpl[object][training_pose][0] == maxSimKNNTmpl[object][training_pose][1]) intra++;
            if (maxSimKNNTmpl[object][training_pose][0] == object) inter++;
        }
    }
    cout << "Intra-class accuracy: " << intra/(float)(nr_objects*nr_training_poses)*100 << endl;
    cout << "Inter-class accuracy: " << inter/(float)(nr_objects*nr_training_poses)*100 << endl;
}




