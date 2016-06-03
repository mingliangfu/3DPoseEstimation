#include "networksolver.h"

networkSolver::networkSolver(vector<string> used_models, string network_path, string hdf5_path, datasetManager db_manager):
               used_models(used_models), network_path(network_path), hdf5_path(hdf5_path), db_manager(db_manager)
{
    // Generate the datasets out of the stored h5 files
    db_manager.generateDatasets(used_models, training_set, test_set, templates);

    // For each object of the dataset
    for (size_t i = 0; i < used_models.size(); ++i)
    {
        // - build a mapping from model name to index number
        model_index[used_models[i]] = i;
    }

    // Read quaternion poses from templates (they are identical for all objects)
    int nr_poses = templates[0].size();
    tmpl_quats.assign(nr_poses, Quaternionf());
    for  (int i=0; i < nr_poses; ++i)
        for (int j=0; j < 4; ++j)
            tmpl_quats[i].coeffs()(j) = templates[0][i].label.at<float>(0,1+j);

    // Shuffle the training set and recalculate the quaternions
    shuffleTrainingSet();
}

void networkSolver::shuffleTrainingSet()
{
    // Shuffle training set
    int nr_objects = used_models.size();
    for (int object = 0; object < nr_objects; ++object) {
        random_shuffle(training_set[object].begin(), training_set[object].end());
    }

    // Read quaternion poses from training data
    training_quats.assign(nr_objects, vector<Quaternionf, Eigen::aligned_allocator<Quaternionf>>());
    for  (int i=0; i < nr_objects; ++i) {
        training_quats[i].resize(training_set[i].size());
        for (size_t k=0; k < training_quats[i].size(); ++k)
            for (int j=0; j < 4; ++j)
                training_quats[i][k].coeffs()(j) = training_set[i][k].label.at<float>(0,1+j);
    }

}

vector<Sample> networkSolver::buildBatch(int batch_size, int iter)
{
    int nr_objects = training_set.size();
    int nr_template_poses = templates[0].size();
    int triplet_size = 5;
    vector<Sample> batch;
    size_t puller, pusher0, pusher1, pusher2;
    TripletWang triplet;

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_template_poses-1);

    for (int linearId = iter * batch_size/triplet_size; linearId < (iter * batch_size/triplet_size) + batch_size/triplet_size; ++linearId) {

        // Calculate 2d indices
        int training_sample = linearId / nr_objects;
        int obj = linearId % nr_objects;

        // Pull random scene sample and find closest pose neighbor from templates
        float best_dist = numeric_limits<float>::max();
        float best_dist2 = numeric_limits<float>::max(); // second best

        // Remember the anchor
        triplet.anchor = training_set[obj][training_sample];

        // Find the puller: most similar template
        for (size_t temp = 0; temp < nr_template_poses; temp++)
        {
            float temp_dist = training_quats[obj][training_sample].angularDistance(tmpl_quats[temp]);
            if (temp_dist >= best_dist) continue;
            puller = temp;
            best_dist = temp_dist;
        }
        triplet.puller = templates[obj][puller];

        // Find pusher0: second most similar template
        for (size_t temp = 0; temp < nr_template_poses; temp++)
        {
            float temp_dist = training_quats[obj][training_sample].angularDistance(tmpl_quats[temp]);
            if (temp_dist >= best_dist2 || temp_dist == best_dist) continue;
            pusher0 = temp;
            best_dist2 = temp_dist;
        }
        triplet.pusher0 = templates[obj][pusher0];

        // Find pusher1: random template
        pusher1 = ran_tpl(ran);
        while (pusher1 == puller && pusher1 == pusher0) pusher1 = ran_tpl(ran);
        triplet.pusher1 = templates[obj][pusher1];

        // Find pusher2: template from another object
        pusher2 = ran_obj(ran);
        while (pusher2 == obj) pusher2 = ran_obj(ran);
        triplet.pusher2 = templates[pusher2][ran_tpl(ran)];

#if 0   // Show triplets
        imshow("anchor",showRGBDPatch(triplet.anchor.data,false));
        imshow("puller",showRGBDPatch(triplet.puller.data,false));
        imshow("pusher0",showRGBDPatch(triplet.pusher0.data,false));
        imshow("pusher1",showRGBDPatch(triplet.pusher1.data,false));
        imshow("pusher2",showRGBDPatch(triplet.pusher2.data,false));
        if (training_sample == 0) waitKey();
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

void networkSolver::trainNet(string net_name, int resume_iter)
{
    int nr_objects = used_models.size();
    int nr_training_poses = training_set[0].size();
    int triplet_size = 5;

    // Build a bool vector for each object that stores if all templates have been used yet
    training_used.assign(nr_objects, vector<bool>(nr_training_poses,false));

    // Set network parameters
    caffe::SolverParameter solver_param;
    solver_param.set_base_lr(0.005);
    solver_param.set_momentum(0.9);
    solver_param.set_weight_decay(0.001);

    solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);

    solver_param.set_stepsize(15000);
    solver_param.set_lr_policy("step");
    solver_param.set_gamma(0.9);

    solver_param.set_snapshot(20000);
    solver_param.set_snapshot_prefix(net_name);

    solver_param.set_display(1);
    solver_param.set_net(network_path + net_name + ".prototxt");
    caffe::SGDSolver<float> *solver = new caffe::SGDSolver<float>(solver_param);

    if (resume_iter > 0) {
        string resume_file = network_path + net_name + "_iter_" + to_string(resume_iter) + ".solverstate";
        solver->Restore(resume_file.c_str());
    }

    // Get network information
    boost::shared_ptr<caffe::Net<float> > net = solver->net();
    caffe::Blob<float>* input_data_layer = net->input_blobs()[0];
    const size_t batch_size = input_data_layer->num();
    const int channels =  input_data_layer->channels();
    const int targetSize = input_data_layer->height();
    const int slice = input_data_layer->height()*input_data_layer->width();
    const int img_size = slice*channels;
    vector<float> data(batch_size*img_size,0), labels(batch_size,0);

    vector<Sample> batch;
    int epoch_iter = nr_objects * nr_training_poses / (batch_size/triplet_size);
    int num_epochs = 50;
    int training_rounds = 3;

    // Perform training
    for (int training_round = 0; training_round < training_rounds; ++training_round)
    {
        for (int epoch = 0; epoch <= num_epochs; epoch++)
        {
            for (int iter = 0; iter < epoch_iter; iter++)
            {
                // Fill current batch
                batch = buildBatch(batch_size, iter);

                // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
                for (size_t i=0; i < batch.size(); ++i)
                {
                    int currImg = i*img_size;
                    for (int ch=0; ch < channels ; ++ch)
                        for (int y = 0; y < targetSize; ++y)
                            for (int x = 0; x < targetSize; ++x) {
                                data[currImg + slice*ch + y*targetSize + x] = batch[i].data.ptr<float>(y)[x*channels + ch];
                                labels[i] = batch[i].label.at<float>(0,0); }
                }
                input_data_layer->set_cpu_data(data.data());
                solver->Step(1);
            }
        }

        // Do bootstraping!!!
        shuffleTrainingSet();
    }
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


void networkSolver::testManifold(string net_name, int resume_iter)
{
    caffe::Net<float> CNN(network_path + net_name + ".prototxt", caffe::TEST);
    CNN.CopyTrainedLayersFrom(network_path + net_name + "_iter_" + to_string(resume_iter) + ".caffemodel");

    Mat DBfeats;
    for (int tmpl = 0; tmpl < templates.size(); ++tmpl) {
        DBfeats.push_back(computeDescriptors(CNN, templates[tmpl]));
    }
    int nr_templates = DBfeats.rows/used_models.size();

    // Visualize for the case where feat_dim is 3D
    viz::Viz3d visualizer("Manifold");
    cv::Mat vizMat(DBfeats.rows,1,CV_32FC3, DBfeats.data);
    cv::Mat colorMat;
    std::uniform_int_distribution<size_t> color(0, 255);
    for (int model = 0; model < used_models.size(); ++model) {
        colorMat.push_back(Mat(nr_templates,1,CV_8UC3,Scalar(color(ran), color(ran), color(ran))));
    }

    viz::WCloud manifold(vizMat,colorMat);
    manifold.setRenderingProperty(viz::POINT_SIZE, 5);
    visualizer.setBackgroundColor(cv::viz::Color::white(), cv::viz::Color::white());
    visualizer.setWindowSize(cv::Size(600, 400));
    visualizer.showWidget("cloud", manifold);
    visualizer.spin();

}

Mat networkSolver::showRGBDPatch(Mat &patch, bool show/*=true*/)
{
    vector<Mat> channels;
    //cv::split((patch+1.f)*0.5f,channels);
    cv::split(patch,channels);

    Mat RGB,D,out(patch.rows,patch.cols*2,CV_32FC3);

    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));

    cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);
    D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));

    if(show) {imshow("R G B D",out); waitKey();}
    return out;
}

void networkSolver::testKNN(string net_name, int resume_iter, vector<string> test_models)
{
    // Load the snapshot
    caffe::Net<float> CNN(network_path + net_name + ".prototxt", caffe::TEST);
    CNN.CopyTrainedLayersFrom(network_path + net_name + "_iter_" + to_string(resume_iter) + ".caffemodel");

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
        uniform_int_distribution<size_t> DBtestGen(0, DBtest.rows);
        int DBtestId = DBtestGen(ran);
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



