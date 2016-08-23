#include "../include/networksolver.h"

namespace sz {

networkSolver::networkSolver(string config, datasetManager* db): db(db), template_set(db->getTemplateSet()), training_set(db->getTrainingSet()),
    test_set(db->getTestSet()), maxSimTmpl(db->getMaxSimTmpl()), config(config)
{
    // Read config parameters
    readParam(config);
}

void networkSolver::buildBatchQueue(size_t batch_size, size_t triplet_size, size_t epoch_iter,
                                    size_t slice, size_t channels, size_t target_size, std::queue<vector<float>> &batch_queue)
{
    vector<Sample> batch;
    size_t img_size = slice * channels;
    vector<float> batch_caffe(batch_size * img_size, 0);
    size_t current_iter;

    while(true)
    {
        // Update the batch_id
        std::unique_lock<std::mutex> lock_id(queue_mutex);
        current_iter = thread_iter;
        thread_iter = (thread_iter < epoch_iter - 1) ? thread_iter + 1 : 0;
        lock_id.unlock();

        // Fill the batch
        batch.clear();
        batch = buildBatch(batch_size, triplet_size, current_iter);

        // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
        for (size_t i = 0; i < batch.size(); ++i)
        {
            for (size_t h = 0; h < target_size; ++h) {
                const float* ptr = batch[i].data.ptr<float>(h);
                for (size_t w = 0; w < target_size; ++w)
                    for (size_t c = 0; c < channels; ++c)
                        batch_caffe[i*img_size + (c * target_size + h) * target_size + w] = *(ptr++);
            }
        }

        // Wait until the size of the queue is less than 10 and push the batch to the queue
        std::unique_lock<std::mutex> lock(queue_mutex);
        cond.wait(lock, [&](){return batch_queue.size() < 10;});
        batch_queue.push(batch_caffe);
        cond.notify_all();
        lock.unlock();
    }
}

vector<Sample> networkSolver::buildBatch(int batch_size, unsigned int triplet_size, int iter)
{
    vector<Sample> batch;
    size_t puller = 0, pusher0 = 0, pusher1 = 0, pusher2 = 0;

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_template_poses-1);

    unordered_map<string, vector<Sample> > hard_negs = db->getHardNegatives();

    for (size_t linearId = iter * batch_size/triplet_size; linearId < (iter * batch_size/triplet_size) + batch_size/triplet_size; ++linearId) {

        Triplet triplet;

        // Calculate 2d indices
        unsigned int training_pose = linearId / nr_objects;
        unsigned int object = linearId % nr_objects;

        // Anchor: training set sample
        triplet.anchor.copySample(training_set[object][training_pose]);
        // Puller: most similar template
        puller = maxSimTmpl[object][training_pose][0];
        triplet.puller.copySample(template_set[object][puller]);

        // Pusher 0: second most similar template
        pusher0 = maxSimTmpl[object][training_pose][1];
        triplet.pusher0.copySample(template_set[object][pusher0]);

        // OR:
        // If we have hard negatives for this model, put it
        if (hard_negs.find(used_models[object]) != hard_negs.end())
        {
            vector<Sample> &negs = hard_negs[used_models[object]];
            std::uniform_int_distribution<size_t> ran_neg(0, negs.size()-1);
            triplet.pusher0.copySample(negs[ran_neg(ran)]);
        }

        if (bootstrapping)
        {
            // Pusher 1: fill either with the missclassified nn or random template of the same class
            unsigned int knn_object = maxSimKNNTmpl[object][training_pose][0];
            unsigned int knn_pose = maxSimKNNTmpl[object][training_pose][1];

            if (knn_object != object || knn_pose != puller) {
                // - missclassified nearest neighbor
                triplet.pusher1.copySample(template_set[knn_object][knn_pose]);
            } else {
                // - random template of the same class
                // -- if model is rotInv or symmetric
                if (rotInv[model_index[used_models[object]]] != 0)
                {
                    // -- randomize until elevation levels of the puller and pusher 1 are different
                    pusher1 = ran_tpl(ran);
                    while(abs(acos(template_set[object][pusher1].getQuat().toRotationMatrix()(2,2)) - acos(template_set[object][puller].getQuat().toRotationMatrix()(2,2))) < 0.2)
                        pusher1 = ran_tpl(ran);
                    triplet.pusher1.copySample(template_set[object][pusher1]);

                    // -- if model is normal
                } else {
                    pusher1 = ran_tpl(ran);
                    while(pusher1 == puller && pusher1 == pusher0) pusher1 = ran_tpl(ran);
                    triplet.pusher1.copySample(template_set[object][pusher1]);
                }
            }

            // Pusher 2: fill either with the missclassified knn or random template of a different class
            if (maxSimKNNTmpl[object][training_pose].size() > 2 && (knn_object != object || knn_pose != puller)) {
                // - missclassified knn of a different object
                knn_object = maxSimKNNTmpl[object][training_pose][2];
                knn_pose = maxSimKNNTmpl[object][training_pose][3];
                triplet.pusher2.copySample(template_set[knn_object][knn_pose]);
            } else {

                    // - template from another object
                    pusher2 = ran_obj(ran);
                    while (pusher2 == object) pusher2 = ran_obj(ran);
                    triplet.pusher2.copySample(template_set[pusher2][ran_tpl(ran)]);
            }
        } else {
            // Pusher 1: random template of the same class
            // - if model is rotInv or symmetric
            if (rotInv[model_index[used_models[object]]] != 0)
            {
                // -- randomize until elevation levels of the puller and pusher 1 are different
                pusher1 = ran_tpl(ran);
                while(abs(acos(template_set[object][pusher1].getQuat().toRotationMatrix()(2,2)) - acos(template_set[object][puller].getQuat().toRotationMatrix()(2,2))) < 0.2)
                    pusher1 = ran_tpl(ran);
                triplet.pusher1.copySample(template_set[object][pusher1]);

                // - if model is normal
            } else {
                pusher1 = ran_tpl(ran);
                while(pusher1 == puller && pusher1 == pusher0) pusher1 = ran_tpl(ran);
                triplet.pusher1.copySample(template_set[object][pusher1]);
            }

            // Pusher 2: random template of a different class
            pusher2 = ran_obj(ran);
            while (pusher2 == object) pusher2 = ran_obj(ran);
            triplet.pusher2.copySample(template_set[pusher2][ran_tpl(ran)]);
        }

        // Fill random backgrounds
        if (random_background != 0) {
            db->randomFill(triplet.anchor.data, random_background);
            db->randomFill(triplet.puller.data, random_background);
            db->randomFill(triplet.pusher0.data, random_background);
            db->randomFill(triplet.pusher1.data, random_background);
            db->randomFill(triplet.pusher2.data, random_background);
        }

        // Store triplet to the batch
        batch.push_back(triplet.anchor);
        batch.push_back(triplet.puller);
        batch.push_back(triplet.pusher0);
        batch.push_back(triplet.pusher1);
        batch.push_back(triplet.pusher2);

#if 1   // Show triplets
        showTriplet(triplet.anchor.data,triplet.puller.data,triplet.pusher0.data,triplet.pusher1.data,triplet.pusher2.data);
#endif

    }
    return batch;
}

vector<Sample> networkSolver::buildBatchClass(int batch_size, unsigned int triplet_size, int iter)
{
    vector<Sample> batch;
    size_t puller = 0, pusher0 = 0, pusher1 = 0, pusher2 = 0;

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_template_poses-1);

    for (size_t linearId = iter * batch_size/triplet_size; linearId < (iter * batch_size/triplet_size) + batch_size/triplet_size; ++linearId) {

        Triplet triplet;

        // Calculate 2d indices
        unsigned int training_pose = linearId / nr_objects;
        unsigned int object = linearId % nr_objects;

        // Anchor: training set sample
        triplet.anchor.copySample(training_set[object][training_pose]);

        // Puller: random template
        puller = ran_tpl(ran);
        triplet.puller.copySample(template_set[object][puller]);

        // Pusher 0: random template of a different class
        pusher0 = ran_obj(ran);
        while (pusher0 == object) pusher0 = ran_obj(ran);
        triplet.pusher0.copySample(template_set[pusher0][ran_tpl(ran)]);

        // Pusher 1: random template of a different class
        pusher1 = ran_obj(ran);
        while(pusher1 == object || pusher1 == pusher0) pusher1 = ran_obj(ran);
        triplet.pusher1.copySample(template_set[pusher1][ran_tpl(ran)]);

        // Pusher 2: random template of a different class
        pusher2 = ran_obj(ran);
        while (pusher2 == object || pusher2 == pusher0 || pusher2 == pusher1) pusher2 = ran_obj(ran);
        triplet.pusher2.copySample(template_set[pusher2][ran_tpl(ran)]);

        if (bootstrapping)
        {
            // Pusher 1: fill with missclassified nn
            unsigned int knn_object = maxSimKNNTmpl[object][training_pose][0];
            if (knn_object != object) {
                triplet.pusher1.copySample(template_set[knn_object][ran_tpl(ran)]);
            }

            // Pusher 2: fill with missclassified 2nd nn
            if (maxSimKNNTmpl[object][training_pose].size() > 2) {
                knn_object = maxSimKNNTmpl[object][training_pose][2];
                triplet.pusher2.copySample(template_set[knn_object][ran_tpl(ran)]);
            }
        }

        // Fill random backgrounds
        if (random_background != 0) {
            db->randomFill(triplet.anchor.data, random_background);
            db->randomFill(triplet.puller.data, random_background);
            db->randomFill(triplet.pusher0.data, random_background);
            db->randomFill(triplet.pusher1.data, random_background);
            db->randomFill(triplet.pusher2.data, random_background);
        }

        // Store triplet to the batch
        batch.push_back(triplet.anchor);
        batch.push_back(triplet.puller);
        batch.push_back(triplet.pusher0);
        batch.push_back(triplet.pusher1);
        batch.push_back(triplet.pusher2);

#if 1   // Show triplets
        showTriplet(triplet.anchor.data,triplet.puller.data,triplet.pusher0.data,triplet.pusher1.data,triplet.pusher2.data);
#endif

    }
    return batch;
}

void networkSolver::trainNet(int resume_iter, bool threaded)
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


    if (binarization){
        cerr << "Binarization!!" << endl;
        solver_param.set_snapshot_prefix(binarization_net_name);
        solver_param.set_net(network_path + binarization_net_name + ".prototxt");
    }

    caffe::SGDSolver<float> solver(solver_param);
    bootstrapping = false;

    // Store the test network
    caffe::Net<float> testCNN(network_path + net_name + ".prototxt", caffe::TEST);

    if (resume_iter > 0) {
        string resume_file = net_name + "_iter_" + to_string(resume_iter) + ".solverstate";
        solver.Restore(resume_file.c_str());
        computeKNN(testCNN);
        bootstrapping = true;
    }

    // Get network information
    boost::shared_ptr<caffe::Net<float> > net = solver.net();
    caffe::Blob<float>* input_data_layer = net->input_blobs()[0];
    const size_t batch_size = input_data_layer->num();
    const int channels =  input_data_layer->channels();
    const int target_size = input_data_layer->height();
    const int slice = input_data_layer->height()*input_data_layer->width();
    const int img_size = slice * channels;

    unsigned int triplet_size = 5;
    unsigned int epoch_iter = nr_objects * nr_training_poses / (batch_size/triplet_size);
    vector<float> batch_caffe(batch_size * slice*channels,0);
    queue<vector<float>> batch_queue;
    vector<std::thread> threads;

    if (threaded)
    {
        // Start threaded batch builders
        thread_iter = 0;
        threads.resize(std::thread::hardware_concurrency()/2);  // - number of threads supported/2
        for(std::thread &t : threads)
            t = std::thread(&networkSolver::buildBatchQueue, this, batch_size, triplet_size, epoch_iter,
                            slice, channels, target_size, std::ref(batch_queue));
    }

    // Perform training
    for (size_t training_round = 0; training_round < num_training_rounds; ++training_round)
    {
        for (size_t epoch = 0; epoch < num_epochs; epoch++)
        {
            for (size_t iter = 0; iter < epoch_iter; iter++)
            {

                if (threaded)
                {
                    // Get a batch from the queue
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    cond.wait(lock, [&](){return !batch_queue.empty();}); // wait until the queue is not empty
                    batch_caffe = batch_queue.front();
                    batch_queue.pop();
                    cond.notify_all();
                    lock.unlock();
                }
                else
                {
                    vector<Sample> batch = buildBatch(batch_size, triplet_size, iter);

                    if (bootstrapping)
                        imwrite("triplet.png",255*showTriplet(batch[0].data,batch[1].data,batch[2].data,batch[3].data,batch[4].data));

                    // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
                    for (size_t i=0; i < batch.size(); ++i)
                    {
                        for (int h = 0; h < target_size; ++h) {
                            const float* ptr = batch[i].data.ptr<float>(h);
                            for (int w = 0; w < target_size; ++w)
                                for (int c = 0; c < channels; ++c)
                                    batch_caffe[i*img_size + (c * target_size + h) * target_size + w] = *(ptr++);
                        }
                    }
                }


                input_data_layer->set_cpu_data(batch_caffe.data());
                solver.Step(1);
            }
        }

        // Do bootstraping
        int snapshot_iter = epoch_iter * num_epochs * (training_round + 1) + resume_iter;
        testCNN.ShareTrainedLayersWith(&(*net));

        if (random_background != 0) {
            vector<vector<Sample>> copy_tmpl(nr_objects, vector<Sample>(nr_template_poses));
            for (size_t object = 0; object < nr_objects; ++object) {
                for (size_t pose = 0; pose < nr_template_poses; ++pose) {
                    copy_tmpl[object][pose].copySample(template_set[object][pose]);
                    db->randomFill(copy_tmpl[object][pose].data, random_background);
                }
            }
            eval::computeHistogram(testCNN, copy_tmpl, training_set, test_set, rotInv, config, snapshot_iter);
            eval::computeHistogram(testCNN, template_set, training_set, test_set, rotInv, config, snapshot_iter);
        } else {
            eval::computeHistogram(testCNN, template_set, training_set, test_set, rotInv, config, snapshot_iter);
        }

        computeKNN(testCNN);
        bootstrapping = true;

    }
    solver.Snapshot();
    clog << "Training finished!" << endl;

    // Detach all threads
    for(std::thread &t : threads)
        t.detach();

}


void networkSolver::binarizeNet(int resume_iter)
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
    solver_param.set_snapshot_prefix(binarization_net_name);
    solver_param.set_display(1);
    solver_param.set_net(network_path + binarization_net_name + ".prototxt");
    caffe::SGDSolver<float> solver(solver_param);

    if (resume_iter > 0) {
        string resume_file = net_name + "_iter_" + to_string(resume_iter) + ".caffemodel";
        solver.net()->CopyTrainedLayersFrom(resume_file.c_str());
    }

    // Get network information
    boost::shared_ptr<caffe::Net<float> > net = solver.net();
    caffe::Blob<float>* input_data_layer = net->input_blobs()[0];
    const size_t batch_size = input_data_layer->num();
    const int channels =  input_data_layer->channels();
    const int targetSize = input_data_layer->height();
    const int slice = input_data_layer->height()*input_data_layer->width();
    const int img_size = slice*channels;
    vector<float> data(batch_size*img_size,0);



    vector<Sample> batch;
    unsigned int triplet_size = 5;
    unsigned int epoch_iter = nr_objects * nr_training_poses / (batch_size/triplet_size);

    // Perform training
    for (size_t epoch = 0; epoch < binarization_epochs; epoch++)
    {
        for (size_t iter = 0; iter < epoch_iter; iter++)
        {
            // Fill current batch
            batch = buildBatch(batch_size, iter, triplet_size);

            // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
            for (size_t i=0; i < batch.size(); ++i)
            {
                int currImg = i*img_size;
                for (int ch = 0; ch < channels ; ++ch)
                    for (int y = 0; y < targetSize; ++y)
                        for (int x = 0; x < targetSize; ++x)
                            data[currImg + slice*ch + y*targetSize + x] = batch[i].data.ptr<float>(y)[x*channels + ch];
            }
            input_data_layer->set_cpu_data(data.data());
            solver.Step(1);
        }
    }

    solver.Snapshot();
    clog << "Binarization training finished!" << endl;
}


void networkSolver::computeKNN(caffe::Net<float> &CNN)
{
    // Get the training data
    Mat DBfeats, DBtraining;
    for (size_t model_id = 0; model_id < used_models.size(); ++model_id) {
        DBtraining.push_back(eval::computeDescriptors(CNN, training_set[model_id]));
        DBfeats.push_back(eval::computeDescriptors(CNN, template_set[model_id]));
    }

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    matcher->add(DBfeats);

    // - match the DBtrainng
    vector< vector<DMatch> > matches;
    int knn = 3;
    matcher->knnMatch(DBtraining, matches, knn);

    maxSimKNNTmpl.assign(nr_objects, vector<vector<int>>(nr_training_poses, vector<int>()));

    for (size_t linearId = 0; linearId < (unsigned)DBtraining.rows; ++linearId) {

        // - get the test set indices (1D -> 2D)
        int query_object = linearId / nr_training_poses;
        int query_pose = linearId % nr_training_poses;

        for (int nn = 0; nn < knn; nn++)
        {
            int tmpl_object =  matches[linearId][nn].trainIdx / nr_template_poses;
            int tmpl_pose =  matches[linearId][nn].trainIdx % nr_template_poses;
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

void networkSolver::readParam(string config)
{
    // Initialize the parser
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);

    // Read learning parameters
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

    binarization = pt.get<bool>("train.binarization");
    binarization_epochs = pt.get<int>("train.binarization_epochs");
    binarization_net_name = pt.get<string>("train.binarization_net_name");
    rotInv = to_array<int>(pt.get<string>("input.rotInv"));
    inplane = pt.get<bool>("input.inplane");
    random_background = pt.get<int>("input.random_background");

    if (gpu) caffe::Caffe::set_mode(caffe::Caffe::GPU);
    used_models = to_array<string>(pt.get<string>("input.used_models"));
    models = to_array<string>(pt.get<string>("input.models"));

    // Save dataset parameters
    nr_objects = used_models.size();
    nr_training_poses = training_set.front().size();
    nr_template_poses = template_set.front().size();
    nr_test_poses = test_set.front().size();

    // For each object build a mapping from model name to index number
    for (size_t i = 0; i < models.size(); ++i) model_index[models[i]] = i;

}
}
