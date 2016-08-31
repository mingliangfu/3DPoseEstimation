#include "../include/networkevaluator.h"

namespace sz {

networkEvaluator::networkEvaluator()
{
}

Mat networkEvaluator::computeDescriptors(caffe::Net<float> &CNN, vector<Sample> samples)
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
                for (int h = 0; h < targetSize; ++h) {
                    const float* ptr = patch.ptr<float>(h);
                    for (int w = 0; w < targetSize; ++w)
                        for (int c = 0; c < channels; ++c)
                            data[j*img_size + (c * targetSize + h) * targetSize + w] = *(ptr++);
                }
            }
            // Copy data memory into Caffe input layer, process batch and copy result back
            input_layer->set_cpu_data(data.data());
            vector< caffe::Blob<float>* > out = CNN.Forward();

            for (size_t j=0; j < currIdxs.size(); ++j)
                memcpy(descs.ptr<float>(currIdxs[j]), out[0]->cpu_data() + j*desc_dim, desc_dim*sizeof(float));

            currIdxs.clear(); // Throw away current batch
        }
    }
    return descs;
}

void networkEvaluator::computeManifold(caffe::Net<float> &CNN, const vector<vector<Sample>> &templates, int iter)
{
    std::random_device ran;
    Mat DBfeats;
    for (size_t tmpl = 0; tmpl < templates.size(); ++tmpl) {
        DBfeats.push_back(computeDescriptors(CNN, templates[tmpl]));
    }
    int nr_templates = DBfeats.rows/templates.size();

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
    for (size_t model = 0; model < templates.size(); ++model) {
        colorMat.push_back(Mat(nr_templates,1,CV_8UC3,Scalar(color(ran), color(ran), color(ran))));
    }

    viz::WCloud manifold(vizMat,colorMat);
    manifold.setRenderingProperty(viz::POINT_SIZE, 5);
    visualizer.setBackgroundColor(cv::viz::Color::white(), cv::viz::Color::white());
    visualizer.setWindowSize(cv::Size(600, 400));
    visualizer.showWidget("cloud", manifold);
    visualizer.spin();
//    visualizer.spinOnce();
    visualizer.saveScreenshot("manifold_" + to_string(iter) + ".png");
}

void networkEvaluator::visualizeKNN(caffe::Net<float> &CNN,
                                    const vector<vector<Sample>> &test_set, const vector<vector<Sample>> &templates)
{
    // Get the test data: subset of the used_models
    Mat DBfeats, DBtest;
    std::random_device ran;

    for (size_t obj = 0; obj < test_set.size(); ++obj) {
        DBtest.push_back(computeDescriptors(CNN, test_set[obj]));
        DBfeats.push_back(computeDescriptors(CNN, templates[obj]));
    }

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    matcher->add(DBfeats);

    // Enter the infinite loop
    while (true)
    {
        // - get the random index for the query object
        uniform_int_distribution<size_t> DBtestGen(0, DBtest.rows - 1);
        int DBtestId = DBtestGen(ran);
        Mat testDescr = DBtest.row(DBtestId).clone();

        // - match the DBtest
        vector< vector<DMatch> > matches;
        int knn = 5;
        matcher->knnMatch(testDescr, matches, knn);

        // - get the test set indices (1D -> 2D)
        int query_object = DBtestId / test_set[0].size();
        int query_sample = DBtestId % test_set[0].size();

        // - show the query object
        imshow("query",showRGBDPatch(test_set[query_object][query_sample].data,false));

        for (DMatch &m : matches[0])
        {
            // - get the template set indices (1D -> 2D)
            int tmpl_object = m.trainIdx / templates[0].size();
            int tmpl_sample = m.trainIdx % templates[0].size();

            // - show the k-NN object
            imshow("kNN",showRGBDPatch(templates[tmpl_object][tmpl_sample].data,false));

            // - get the quaternions of the compared objects
            Quaternionf queryQuat, kNNQuat;
            for (int i = 0; i < 4; ++i)
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

void networkEvaluator::computeKNNAccuracy(vector<vector<vector<int>>> &maxSimTmpl, vector<vector<vector<int>>> &maxSimKNNTmpl)
{
    int intra = 0, inter = 0;
    int nr_objects = maxSimTmpl.size();
    int nr_training_poses = maxSimTmpl[0].size();
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


vector<vector<float>> networkEvaluator::computeConfusionMatrix(caffe::Net<float> &CNN,
                                             const vector<vector<Sample>> &template_set,
                                             const vector<vector<Sample>> &test_set, vector<string> models, unordered_map<string, int> local_index, int knn)
{
    // Get the test data
    Mat DBfeats, DBtest;
    int nr_test_poses = test_set[0].size();
    int nr_tmpl_poses = template_set[0].size();

    for (size_t object = 0; object < template_set.size(); ++object) {
        DBfeats.push_back(computeDescriptors(CNN, template_set[object]));
        DBtest.push_back(computeDescriptors(CNN, test_set[object]));
    }

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher;
    vector< vector<DMatch> > matches;
    matcher = DescriptorMatcher::create("BruteForce");
    matcher->add(DBfeats);
    matcher->knnMatch(DBtest, matches, knn);

    vector<vector<float>> conf_matrix(local_index.size(), vector<float>(local_index.size(), 0.0f));

    for (size_t linearId = 0; linearId < (unsigned)DBtest.rows; ++linearId) {

        // - get the test set indices (1D -> 2D)
        int query_object = linearId / nr_test_poses;
        int query_pose = linearId % nr_test_poses;

        for (int nn = 0; nn < knn; nn++)
        {
            int knn_object =  matches[linearId][nn].trainIdx / nr_tmpl_poses;
            int knn_pose =  matches[linearId][nn].trainIdx % nr_tmpl_poses;

            if (test_set[query_object][query_pose].label.at<float>(0,0) == template_set[knn_object][knn_pose].label.at<float>(0,0))
            {
                // -- fill confusion matrix
                int local_query_label = local_index[models[test_set[query_object][query_pose].label.at<float>(0,0)]];
                int local_knn_label = local_index[models[template_set[knn_object][knn_pose].label.at<float>(0,0)]];
                conf_matrix[local_query_label][local_knn_label]++;
                break;
            }

            // - if occurance of the same class is not found among kNNs fill in the first one
            if (nn == knn-1)
            {
                // -- get the 1st nn
                knn_object =  matches[linearId][0].trainIdx / nr_tmpl_poses;
                knn_pose =  matches[linearId][0].trainIdx % nr_tmpl_poses;
                // -- fill confusion matrix
                int local_query_label = local_index[models[test_set[query_object][query_pose].label.at<float>(0,0)]];
                int local_knn_label = local_index[models[template_set[knn_object][knn_pose].label.at<float>(0,0)]];
                conf_matrix[local_query_label][local_knn_label]++;
            }
#if 0
            imshow("query",showRGBDPatch(test_set[query_object][query_pose].data,false));
            imshow("KNN",showRGBDPatch(template_set[knn_object][knn_pose].data,false));
            waitKey();
#endif
        }

    }

    // Normalize
    for (vector<float> &v : conf_matrix) {
       for (float &x : v) x /= nr_test_poses;
    }
    return conf_matrix;
}

vector<float> networkEvaluator::computeHistogram(caffe::Net<float> &CNN,
                                        const vector<vector<Sample>> &template_set,
                                        const vector<vector<Sample>> &test_set, vector<int> rotInv, vector<float> bins, int knn)
{
    // Get the test data
     Mat DBfeats, DBtest;
     int nr_test_poses = test_set[0].size();
     int nr_tmpl_poses = template_set[0].size();

     for (size_t object = 0; object < template_set.size(); ++object) {
         DBfeats.push_back(computeDescriptors(CNN, template_set[object]));
         DBtest.push_back(computeDescriptors(CNN, test_set[object]));
     }

     // Create a k-NN matcher
     cv::Ptr<DescriptorMatcher> matcher;
     vector< vector<DMatch> > matches;
     matcher = DescriptorMatcher::create("BruteForce");
     matcher->add(DBfeats);
     matcher->knnMatch(DBtest, matches, knn);

     vector<float> histo(bins.size(), 0);
     float median_angle, mean_angle = 0;
     vector<float> diff_vector;

     for (size_t linearId = 0; linearId < (unsigned)DBtest.rows; ++linearId) {

         // - get the test set indices (1D -> 2D)
         int query_object = linearId / nr_test_poses;
         int query_pose = linearId % nr_test_poses;
         Quaternionf query_quat = test_set[query_object][query_pose].getQuat();

         // - loop through nns and choose the best one
         float best_dist = numeric_limits<float>::max();
         int best_knn_object = -1, best_knn_pose;
         for (int nn = 0; nn < knn; nn++)
         {
             int knn_object =  matches[linearId][nn].trainIdx / nr_tmpl_poses;
             int knn_pose =  matches[linearId][nn].trainIdx % nr_tmpl_poses;

             if (test_set[query_object][query_pose].label.at<float>(0,0) == template_set[knn_object][knn_pose].label.at<float>(0,0))
             {
                 // Get the quaternions
                 Quaternionf knn_quat = template_set[knn_object][knn_pose].getQuat();

                 // Angular difference
                 float knn_dist;
                 // If object is normal compare angular distance, else elevation level
                 if(rotInv[test_set[query_object][query_pose].label.at<float>(0,0)] == 0) {
                     knn_dist = query_quat.angularDistance(knn_quat)*180.f/M_PI;
                 } else {
                     knn_dist = abs(acos(query_quat.toRotationMatrix()(2,2)) - acos(knn_quat.toRotationMatrix()(2,2)))*180.f/M_PI;
                     if(isnan(knn_dist)) knn_dist = numeric_limits<float>::max();
                 }

                 if (knn_dist >= best_dist) continue;
                 best_dist = knn_dist;
                 best_knn_object = knn_object;
                 best_knn_pose = knn_pose;
             }
         }

         // If objects are different fill the first bin
         if (best_knn_object == -1) {
             histo[0]++; continue;
         }
#if 0
         imshow("query",showRGBDPatch(test_set[query_object][query_pose].data,false));
         imshow("KNN",showRGBDPatch(template_set[best_knn_object][best_knn_pose].data,false));
         waitKey();
#endif

         // Store the angular differences
         mean_angle += best_dist;
         diff_vector.push_back(best_dist);


         // Exact match?
         bool exact_match = true;
         Quaternionf best_knn_quat = template_set[best_knn_object][best_knn_pose].getQuat();

         for (auto tmpl_pose = 0; tmpl_pose < nr_tmpl_poses; ++tmpl_pose) {
             Quaternionf tmpl_quat = template_set[best_knn_object][tmpl_pose].getQuat();
             if (query_quat.angularDistance(tmpl_quat) < query_quat.angularDistance(best_knn_quat))
                 exact_match = false;
         }
         if (exact_match) histo[1]++;

         // Fill histogram angles
         for (size_t b = 2; b < bins.size(); ++b)
             if (best_dist < bins[b]) histo[b]++;
     }

     // Normalize histogram values
     float total = histo.front() + histo.back();
     for (float &i : histo) i /= total;

     // Get mean angular difference
     mean_angle /= DBtest.rows;

     // Get median angular difference
     sort(diff_vector.begin(), diff_vector.end());
     median_angle = diff_vector[(diff_vector.size()-1)/2];

     histo.push_back(mean_angle);
     histo.push_back(median_angle);

     return histo;
}

void networkEvaluator::saveConfusionMatrix(caffe::Net<float> &CNN, datasetManager &db, string config)
{
    // Initialize the parser
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);

    // - read config parameters
    int knn = pt.get<int>("output.kNN");
    int random_background = pt.get<int>("input.random_background");
    vector<string> models = to_array<string>(pt.get<string>("input.models"));
    vector<string> used_models = to_array<string>(pt.get<string>("input.used_models"));
    unordered_map<string, int> local_index;
    for (size_t i = 0; i < used_models.size(); ++i) local_index[used_models[i]] = i;

    // - get the test data
    const vector<vector<Sample>>& template_set = db.getTemplateSet();
    const vector<vector<Sample>>& test_set = db.getTestSet();
    vector<vector<float>> conf_matrix;

    if (random_background != 0) {
        // -- fill template backgrounds
        vector<vector<Sample>> template_set_rb(template_set.size(), vector<Sample>(template_set[0].size()));
        for (size_t object = 0; object < template_set.size(); ++object) {
            for (size_t pose = 0; pose < template_set[0].size(); ++pose) {
                template_set_rb[object][pose].copySample(template_set[object][pose]);
                db.randomFill(template_set_rb[object][pose].data, random_background);
            }
        }
        // -- compute histogram for test/train data
        conf_matrix = computeConfusionMatrix(CNN, template_set_rb, test_set, models, local_index, knn);
    } else {
        // -- compute histogram for test/train data
        conf_matrix = computeConfusionMatrix(CNN, template_set, db.getTrainingSet(), models, local_index, knn);
    }

    // Write stats to the file
    // - read learning parameters
    string log_name = pt.get<string>("output.log_name");
    string output_path = pt.get<string>("paths.output_path");

    ofstream log_file;
    log_file.open(output_path + "cm" + log_name);

    // - print the header (once)
    for (size_t model = 0; model < used_models.size()-1; model++)
        log_file << used_models[model] << "\t";
    log_file << used_models.back() << endl;

    // - print the stats - test
    for (size_t model = 0; model < used_models.size(); model++) {
        log_file << used_models[model] << "\t";
        for (size_t value = 0; value < used_models.size()-1; value++)
            log_file << std::setprecision(3) << conf_matrix[model][value]*100 << "\t";
        log_file << std::setprecision(3) << conf_matrix[model].back()*100 << endl;
    }
    log_file.close();
}

void networkEvaluator::saveLog(caffe::Net<float> &CNN, datasetManager &db, string config, int iter)
{
    // Initialize the parser
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config, pt);

    // - read config parameters
    int knn = pt.get<int>("output.kNN");
    vector<int> rotInv = to_array<int>(pt.get<string>("input.rotInv"));
    int random_background = pt.get<int>("input.random_background");

    // - get the test data
    const vector<vector<Sample>>& template_set = db.getTemplateSet();
    const vector<vector<Sample>>& training_set = db.getTrainingSet();
    const vector<vector<Sample>>& test_set = db.getTestSet();
    vector<float> test_hist, train_hist;
    vector<float> bins = {-1, 0, 10, 20, 40, 180};

    if (random_background != 0) {
        // -- fill template backgrounds
        vector<vector<Sample>> template_set_rb(template_set.size(), vector<Sample>(template_set[0].size()));
        for (size_t object = 0; object < template_set.size(); ++object) {
            for (size_t pose = 0; pose < template_set[0].size(); ++pose) {
                template_set_rb[object][pose].copySample(template_set[object][pose]);
                db.randomFill(template_set_rb[object][pose].data, random_background);
            }
        }
        // -- compute histogram for test/train data
        test_hist = computeHistogram(CNN, template_set_rb, test_set, rotInv, bins, knn);
        train_hist = computeHistogram(CNN, template_set_rb, training_set, rotInv, bins, knn);
    } else {
        // -- compute histogram for test/train data
        test_hist = computeHistogram(CNN, template_set, test_set, rotInv, bins, knn);
        train_hist = computeHistogram(CNN, template_set, training_set, rotInv, bins, knn);
    }

    // Write stats to the file
    // - read learning parameters
    string log_name = pt.get<string>("output.log_name");
    string output_path = pt.get<string>("paths.output_path");
    vector<string> used_models = to_array<string>(pt.get<string>("input.used_models"));
    int epoch = iter/(used_models.size() * training_set[0].size() / 60);

    ofstream log_file; ifstream log_file_check;
    log_file.open(output_path + "log" + log_name, ios::app);
    log_file_check.open(output_path + "log" + log_name);

    // - print the header (once)
    if (log_file_check.peek() == std::fstream::traits_type::eof())
    {
        log_file << "mod" << "\t" << "iter" << "\t" << "epoch" << "\t";
        for (size_t i = 0; i < bins.size(); ++i)
            log_file << "<" << bins[i] << "\t";
        log_file << "mean" << "\t" << "median" << endl;
    }

    // - print the stats - test
    log_file << "test" << "\t" << iter << "\t" << epoch << "\t";
    for (size_t i = 0; i < bins.size(); ++i)
        log_file << std::setprecision(3) << test_hist[i]*100 << "\t";
    log_file << test_hist[test_hist.size()-2] << "\t" << test_hist.back() << endl;

    // - print the stats - train
    log_file << "train" << "\t" << iter << "\t" << epoch << "\t";
    for (size_t i = 0; i < bins.size(); ++i)
        log_file << std::setprecision(3) << train_hist[i]*100 << "\t";
    log_file << train_hist[train_hist.size()-2] << "\t" << train_hist.back() << endl;

    log_file.close();

}

}
