#include "networksolver.h"

networkSolver::networkSolver(string network_path, string hdf5_path):
               network_path(network_path), hdf5_path(hdf5_path)
{
}

TripletsPairs networkSolver::buildTripletsPairs(vector<string> used_models)
{
    vector<Triplet> triplets;
    vector<Pair> pairs;
    TripletsPairs triplets_pairs;

    size_t nr_objects = used_models.size();
    assert(nr_objects > 1);

    // Read stored templates and scene samples for the used models
    vector< vector<Sample> > training, templates, temp;
    for (string &seq : used_models)
    {
        training.push_back(h5.read(hdf5_path + "realSamples_" + seq +".h5"));
        temp.push_back(h5.read(hdf5_path + "synthSamples_" + seq +".h5"));
        training[training.size()-1].insert(training[training.size()-1].end(), temp[0].begin(), temp[0].end());
        temp.clear();
        templates.push_back(h5.read(hdf5_path + "templates_" + seq + ".h5"));


    }

    // Read quaternion poses from training data
    vector< vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > > training_quats(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
    {
        training_quats[i].resize(training[i].size());
        for (size_t k=0; k < training_quats[i].size(); ++k)
            for (int j=0; j < 4; ++j)
                training_quats[i][k].coeffs()(j) = training[i][k].label.at<float>(0,1+j);
    }

    // Read quaternion poses from templates (they are identical for all objects)
    int nr_poses = templates[0].size();
    vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > tmpl_quats(nr_poses);
    for  (int i=0; i < nr_poses; ++i)
        for (int j=0; j < 4; ++j)
            tmpl_quats[i].coeffs()(j) = templates[0][i].label.at<float>(0,1+j);

    // Build a bool vector for each object that stores if all templates have been used yet
    vector< vector<bool> > tmpl_used(nr_objects);
    for (auto &vec : tmpl_used)
        vec.assign(nr_poses,false);

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_poses-1);

    // Random generators that returns a random scene sample for a given object
    vector< std::uniform_int_distribution<size_t> > ran_training_sample(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
        ran_training_sample[i] = std::uniform_int_distribution<size_t>(0, training[i].size()-1);


    for(bool finished=false; !finished;)
    {
        size_t anchor, puller, pusher, oldpusher;

        // Build each triplet by cycling through each object
        for (size_t obj=0; obj < nr_objects; obj++)
        {

            /// Type 0: A random scene sample together with closest template against another template

            // Pull random scene sample and find closest pose neighbor from templates
            size_t ran_sample = ran_training_sample[obj](ran);
            puller=0;
            float best_dist = numeric_limits<float>::max();
            for (size_t temp=0; temp < nr_poses; temp++)
            {
//                if (tmpl_used[obj][temp]) continue;  // Skip if template already used
                float temp_dist = training_quats[obj][ran_sample].angularDistance(tmpl_quats[temp]);
                if (temp_dist >= best_dist) continue;
                puller = temp;
                best_dist = temp_dist;
            }


            // Mark template as used
//            tpml_used[obj][puller] = true;


            // Randomize through until pusher != puller
            pusher = ran_tpl(ran);
            while (pusher == puller) pusher = ran_tpl(ran);

            Triplet triplet0;
            triplet0.anchor = training[obj][ran_sample];
            triplet0.puller = templates[obj][puller];
            triplet0.pusher = templates[obj][pusher];
            triplets.push_back(triplet0);
            oldpusher = pusher;


            /// Type 1: All templates are from same object but first two are closer than the third

            // Randomize through until pusher is neither anchor nor puller
            pusher = ran_tpl(ran);
            while ((pusher == puller) && (pusher == oldpusher)) pusher = ran_tpl(ran);

            Triplet triplet1;
            triplet1.anchor = training[obj][ran_sample];
            triplet1.puller = templates[obj][puller];
            triplet1.pusher = templates[obj][pusher];
            triplets.push_back(triplet1);

            /// Type 2: two templates are from same object, the third from another

            // Randomize through until pusher is another object
            pusher = ran_obj(ran);
            while (pusher == obj) pusher = ran_obj(ran);

            Triplet triplet2;
            triplet2.anchor = training[obj][ran_sample];
            triplet2.puller = templates[obj][puller];
            triplet2.pusher = templates[pusher][ran_tpl(ran)];
            triplets.push_back(triplet2);


#if 0      // Show triplets
            for (size_t idx = triplets.size()-3; idx < triplets.size(); idx++)
            {
                imshow("anchor",showRGBDPatch(triplets[idx].anchor.data,false));
                imshow("puller",showRGBDPatch(triplets[idx].puller.data,false));
                imshow("pusher",showRGBDPatch(triplets[idx].pusher.data,false));
                waitKey();
            }

#endif

            /// Add pairs
            Pair pair;
            pair.anchor = training[obj][ran_sample];
            pair.puller = templates[obj][puller];
            pairs.push_back(pair);

        }

        // Check if we are finished (if all templates of all objects were anchors once)
        for (auto &vec : tmpl_used)
            for (int i=0; i < nr_poses; ++i) finished &= vec[i];

        finished = triplets.size()>100000;

    }
    triplets_pairs.triplets = triplets;
    triplets_pairs.pairs = pairs;

    return triplets_pairs;
}

vector<TripletWang> networkSolver::buildTripletsWang(vector<string> used_models)
{
    vector<TripletWang> triplets;

    size_t nr_objects = used_models.size();
    assert(nr_objects > 1);

    // Read stored templates and scene samples for the used models
    vector<vector<Sample>> training, templates;
    for (string &seq : used_models)
    {
        vector<Sample> temp_real(h5.read(hdf5_path + "realSamples_" + seq + ".h5"));
        vector<Sample> temp_synth(h5.read(hdf5_path + "synthSamples_" + seq + ".h5"));

        // Crop the vector: real 50%, synthetic 70%
        temp_real.resize((temp_real.size()-1)*0.5);
        temp_synth.resize((temp_synth.size()-1)*0.7);

        vector<Sample> temp_sum(temp_real);
        temp_sum.insert(temp_sum.end(), temp_synth.begin(), temp_synth.end());

        training.push_back(temp_sum);
        templates.push_back(h5.read(hdf5_path + "templates_" + seq + ".h5"));
    }

    // Read quaternion poses from training data
    vector< vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > > training_quats(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
    {
        training_quats[i].resize(training[i].size());
        for (size_t k=0; k < training_quats[i].size(); ++k)
            for (int j=0; j < 4; ++j)
                training_quats[i][k].coeffs()(j) = training[i][k].label.at<float>(0,1+j);
    }

    // Read quaternion poses from templates (they are identical for all objects)
    int nr_poses = templates[0].size();
    vector<Quaternionf, Eigen::aligned_allocator<Quaternionf> > tmpl_quats(nr_poses);
    for  (int i=0; i < nr_poses; ++i)
        for (int j=0; j < 4; ++j)
            tmpl_quats[i].coeffs()(j) = templates[0][i].label.at<float>(0,1+j);

    // Build a bool vector for each object that stores if all templates have been used yet
    vector< vector<bool> > tmpl_used(nr_objects);
    for (auto &vec : tmpl_used)
        vec.assign(nr_poses,false);

    // Random generator for object selection and template selection
    std::uniform_int_distribution<size_t> ran_obj(0, nr_objects-1), ran_tpl(0, nr_poses-1);

    // Random generators that returns a random scene sample for a given object
    vector< std::uniform_int_distribution<size_t> > ran_training_sample(nr_objects);
    for  (int i=0; i < nr_objects; ++i)
        ran_training_sample[i] = std::uniform_int_distribution<size_t>(0, training[i].size()-1);


    for(bool finished=false; !finished;)
    {
        size_t anchor, puller, pusher0, pusher1, pusher2;
        TripletWang triplet;

        // Build each triplet by cycling through each object
        for (size_t obj=0; obj < nr_objects; obj++)
        {

            // Pull random scene sample and find closest pose neighbor from templates
            size_t ran_sample = ran_training_sample[obj](ran);
            float best_dist = numeric_limits<float>::max();
            float best_dist2 = numeric_limits<float>::max(); // second best

            triplet.anchor = training[obj][ran_sample];

            // Find the puller: most similar template
            for (size_t temp = 0; temp < nr_poses; temp++)
            {
                float temp_dist = training_quats[obj][ran_sample].angularDistance(tmpl_quats[temp]);
                if (temp_dist >= best_dist) continue;
                puller = temp;
                best_dist = temp_dist;
            }
            triplet.puller = templates[obj][puller];

            // Find pusher0: second most similar template
            for (size_t temp = 0; temp < nr_poses; temp++)
            {
                float temp_dist = training_quats[obj][ran_sample].angularDistance(tmpl_quats[temp]);
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
            triplets.push_back(triplet);

#if 0      // Show triplets
            for (size_t idx = triplets.size()-3; idx < triplets.size(); idx++)
            {
                imshow("anchor",showRGBDPatch(triplets[idx].anchor.data,false));
                imshow("puller",showRGBDPatch(triplets[idx].puller.data,false));
                imshow("pusher0",showRGBDPatch(triplets[idx].pusher0.data,false));
                imshow("pusher1",showRGBDPatch(triplets[idx].pusher1.data,false));
                imshow("pusher2",showRGBDPatch(triplets[idx].pusher2.data,false));
                waitKey();
            }
#endif
        }

        finished = triplets.size()>100000;
    }
    return triplets;
}

void networkSolver::trainNet(vector<string> used_models, string net_name, int resume_iter/*=0*/)
{
    caffe::SolverParameter solver_param;
    solver_param.set_base_lr(0.0001);
    solver_param.set_momentum(0.9);
    solver_param.set_weight_decay(0.0005);

    solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);

    solver_param.set_stepsize(1000);
    solver_param.set_lr_policy("step");
    solver_param.set_gamma(0.9);

    int max_iters = 150000;
    solver_param.set_max_iter(150000);

    solver_param.set_snapshot(20000);
    solver_param.set_snapshot_prefix(net_name);

    solver_param.set_display(1);
    solver_param.set_net(network_path + net_name + ".prototxt");
    caffe::SGDSolver<float> *solver = new caffe::SGDSolver<float>(solver_param);
    if (resume_iter>0)
    {
        string resume_file = network_path + net_name + "_iter_" + to_string(resume_iter) + ".solverstate";
        solver->Restore(resume_file.c_str());
    }

    // Read the scene samples
    vector<Sample> batch;
    TripletsPairs triplets_pairs;
    triplets_pairs = buildTripletsPairs(used_models);

    // Get network information
    boost::shared_ptr<caffe::Net<float> > net = solver->net();
    caffe::Blob<float>* input_data_layer = net->input_blobs()[0];
    caffe::Blob<float>* input_label_layer = net->input_blobs()[1];
    const size_t batchSize = input_data_layer->num();
    const int channels =  input_data_layer->channels();
    const int targetSize = input_data_layer->height();
    const int slice = input_data_layer->height()*input_data_layer->width();
    const int img_size = slice*channels;

    vector<float> data(batchSize*img_size,0), labels(batchSize,0);

    // Perform training
    for (int iter = 0; iter <= max_iters; iter++)
    {
        // Fill current batch
        int batch_pairs = 132;
        int batch_triplets = 198;
        batch.clear();

        // Loop through the number of samples per batch
        for (int sampleId = iter*batchSize; sampleId < iter*batchSize; ++sampleId)
        {
            // - fill triplets [:198]
            for (int tripletId = (sampleId-(batch_pairs*iter))/3; tripletId < (sampleId-(batch_pairs*iter))/3 + batch_triplets/3; ++tripletId) {
                batch.push_back(triplets_pairs.triplets[tripletId].anchor);
                batch.push_back(triplets_pairs.triplets[tripletId].puller);
                batch.push_back(triplets_pairs.triplets[tripletId].pusher);
            }
            // - fill pairs [198:330]
            for (int pairId = (sampleId-(batch_triplets*iter))/2; pairId < (sampleId-(batch_triplets*iter))/2 + batch_pairs/2; ++pairId) {
                batch.push_back(triplets_pairs.pairs[pairId].anchor);
                batch.push_back(triplets_pairs.pairs[pairId].puller);
            }
        }

        // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
        for (size_t i=0; i < batch.size(); ++i)
        {
            int currImg = i*img_size;
            for (int ch=0; ch < channels ; ++ch)
                for (int y = 0; y < targetSize; ++y)
                    for (int x = 0; x < targetSize; ++x)
                        data[currImg + slice*ch + y*targetSize + x] = batch[i].data.ptr<float>(y)[x*channels + ch];
            labels[i] = batch[i].label.at<float>(0,0);
        }
        input_data_layer->set_cpu_data(data.data());
        //input_label_layer->set_cpu_data(labels.data());
        solver->Step(1);
    }
}

void networkSolver::trainNetWang(vector<string> used_models, string net_name, int resume_iter)
{
    caffe::SolverParameter solver_param;
    solver_param.set_base_lr(0.0001);
    solver_param.set_momentum(0.9);
    solver_param.set_weight_decay(0.0005);

    solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);

    solver_param.set_stepsize(1000);
    solver_param.set_lr_policy("step");
    solver_param.set_gamma(0.9);

    int max_iters = 150000;
    solver_param.set_max_iter(150000);

    solver_param.set_snapshot(20000);
    solver_param.set_snapshot_prefix(net_name);

    solver_param.set_display(1);
    solver_param.set_net(network_path + net_name + ".prototxt");
    caffe::SGDSolver<float> *solver = new caffe::SGDSolver<float>(solver_param);
    if (resume_iter>0)
    {
        string resume_file = network_path + net_name + "_iter_" + to_string(resume_iter) + ".solverstate";
        solver->Restore(resume_file.c_str());
    }

    // Read the scene samples
    vector<Sample> batch;
    vector<TripletWang> triplets;
    triplets = buildTripletsWang(used_models);

    // Get network information
    boost::shared_ptr<caffe::Net<float> > net = solver->net();
    caffe::Blob<float>* input_data_layer = net->input_blobs()[0];
//    caffe::Blob<float>* input_label_layer = net->input_blobs()[1];
    const size_t batchSize = input_data_layer->num();
    const int channels =  input_data_layer->channels();
    const int targetSize = input_data_layer->height();
    const int slice = input_data_layer->height()*input_data_layer->width();
    const int img_size = slice*channels;

    vector<float> data(batchSize*img_size,0), labels(batchSize,0);

    // Perform training
    for (int iter = 0; iter <= max_iters; iter++)
    {
        // Fill current batch
        batch.clear();

        // Loop through the number of samples per batch
        for (int tripletId = (iter*batchSize)/5; tripletId < (iter*batchSize)/5 + batchSize/5; ++tripletId) {
            batch.push_back(triplets[tripletId].anchor);
            batch.push_back(triplets[tripletId].puller);
            batch.push_back(triplets[tripletId].pusher0);
            batch.push_back(triplets[tripletId].pusher1);
            batch.push_back(triplets[tripletId].pusher2);
        }

        // Fill linear batch memory with input data in Caffe layout with channel-first and set as network input
        for (size_t i=0; i < batch.size(); ++i)
        {
            int currImg = i*img_size;
            for (int ch=0; ch < channels ; ++ch)
                for (int y = 0; y < targetSize; ++y)
                    for (int x = 0; x < targetSize; ++x){
                        data[currImg + slice*ch + y*targetSize + x] = batch[i].data.ptr<float>(y)[x*channels + ch];
                        labels[i] = batch[i].label.at<float>(0,0); }
        }
        input_data_layer->set_cpu_data(data.data());
//        input_label_layer->set_cpu_data(labels.data());
        solver->Step(1);
    }
}

Mat networkSolver::computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples)
{

    caffe::Blob<float>* input_layer = CNN.input_blobs()[0];
    const size_t batchSize = input_layer->num();
    const int channels =  input_layer->channels();
    const int targetSize = input_layer->height();
    const int slice = input_layer->height()*input_layer->width();
    const int img_size = slice*channels;

    caffe::Blob<float>* output_layer = CNN.output_blobs()[0];
    const size_t desc_dim = output_layer->channels();

    vector<float> data(batchSize*img_size,0);
    vector<int> currIdxs;
    Mat descs(samples.size(),desc_dim,CV_32F);

    currIdxs.reserve(batchSize);
    for (size_t i=0; i < samples.size(); ++i)
    {
        // Collect indices of samples to be processed for this batch
        currIdxs.push_back(i);
        if (currIdxs.size() == batchSize || i == samples.size()-1) // If batch full or last sample
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

void networkSolver::testNet()
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    caffe::Net<float> CNN(network_path + "manifold_test.prototxt", caffe::TEST);
    CNN.CopyTrainedLayersFrom(network_path + "manifold_iter_25000.caffemodel");

    vector<Sample> ape = h5.read(hdf5_path + "templates_ape.h5");
    ape.resize(301);
    Mat ape_descs = computeDescriptors(CNN,ape);


    vector<Sample> driller = h5.read(hdf5_path + "templates_driller.h5");
    driller.resize(301);
    Mat driller_descs = computeDescriptors(CNN,driller);

    Mat descs;
    descs.push_back(ape_descs);
    descs.push_back(driller_descs);

    cerr << descs << endl;

    // Visualize for the case where feat_dim is 3D
    viz::Viz3d lol;
    cv::Mat vizMat(descs.rows,1,CV_32FC3, descs.data);
    cv::Mat colorMat;
    colorMat.push_back(Mat(301,1,CV_8UC3,Scalar(255,0,0)));
    colorMat.push_back(Mat(301,1,CV_8UC3,Scalar(0,255,0)));

    lol.showWidget("cloud",viz::WCloud(vizMat,colorMat));
    //lol.setViewerPose(cv::Affine3f::Identity());
    lol.spin();

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

void networkSolver::testKNN(vector<string> used_models)
{

    // Load the snapshot
    caffe::Net<float> CNN(network_path + "manifold_wang.prototxt", caffe::TEST);
    CNN.CopyTrainedLayersFrom(network_path + "manifold_iter_25000.caffemodel");

    // Get the test data
    Mat DBfeats, DBtest;
    vector<Sample> temp_sum, templates;
    for (string &seq : used_models)
    {
        vector<Sample> temp_real(h5.read(hdf5_path + "realSamples_" + seq + ".h5"));
        vector<Sample> temp_synth(h5.read(hdf5_path + "synthSamples_" + seq + ".h5"));
        vector<Sample> temp_tmpl(h5.read(hdf5_path + "templates_" + seq + ".h5"));

        // Crop the vector: real 50%, synthetic 70%
        temp_real.erase(temp_real.begin(),temp_real.begin()+(temp_real.size()-1)*0.5);
        temp_synth.erase(temp_synth.begin(),temp_synth.begin()+(temp_synth.size()-1)*0.7);

        temp_sum.insert(temp_sum.end(), temp_real.begin(), temp_real.end());
        temp_sum.insert(temp_sum.end(), temp_synth.begin(), temp_synth.end());

        templates.insert(templates.end(), temp_tmpl.begin(), temp_tmpl.end());

    }
    DBtest.push_back(computeDescriptors(CNN, temp_sum));
    DBfeats.push_back(computeDescriptors(CNN, templates));

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    matcher->add(DBfeats);

    // Enter the infinite loop
    while(true)
    {
        uniform_int_distribution<size_t> DBtestGen(0, DBtest.rows);
        int DBtestId = DBtestGen(ran);
        Mat testDescr = DBtest.row(DBtestId).clone();

        // - match the DBtest
        vector< vector<DMatch> > matches;
        int knn=5;
        matcher->knnMatch(testDescr, matches, knn);

        // - show the result
        imshow("sceneSample",showRGBDPatch(temp_sum[DBtestId].data,false));

        for (DMatch &m : matches[0])
        {
                imshow("kNN",showRGBDPatch(templates[m.trainIdx].data,false));

//                Quaternionf sampleQuat, kNNQuat;
//                for (int i=0; i < 4; ++i)
//                {
//                    sampleQuat.coeffs()(i) = samples[0].label.at<float>(0,1+i);
//                    kNNQuat.coeffs()(i) = DBfeats[m.trainIdx].label.at<float>(0,1+i);
//                }

//            // -- compute angle difference
//            cout << "Angle difference: " << sampleQuat.angularDistance(kNNQuat) << endl;
            waitKey();
        }
    }
}



