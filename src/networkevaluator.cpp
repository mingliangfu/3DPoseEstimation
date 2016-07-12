#include "../include/networkevaluator.h"

networkEvaluator::networkEvaluator()
{
}

Mat networkEvaluator::computeDescriptors(caffe::Net<float> &CNN, vector<Gopnik::Sample> samples)
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

void networkEvaluator::visualizeManifold(caffe::Net<float> &CNN, const vector<vector<Gopnik::Sample>> &templates, int iter)
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
//    visualizer.spin();
    visualizer.spinOnce();
    visualizer.saveScreenshot("manifold_" + to_string(iter) + ".png");
}

void networkEvaluator::visualizeKNN(caffe::Net<float> &CNN,
                                    const vector<vector<Gopnik::Sample>> &test_set, const vector<vector<Gopnik::Sample>> &templates)
{
    // Get the test data: subset of the used_models
    Mat DBfeats, DBtest;
    std::random_device ran;

    for (int obj = 0; obj < test_set.size(); ++obj) {
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

void networkEvaluator::computeHistogram(caffe::Net<float> &CNN,
                                        const vector<vector<Gopnik::Sample>> &templates,
                                        const vector<vector<Gopnik::Sample>> &training_set,
                                        const vector<vector<Gopnik::Sample>> &test_set, vector<int> rotInv, string config, int iter)
{
    // Get the test data
     Mat DBfeats, DBtest;
     int nr_test_poses = test_set[0].size();
     int nr_tmpl_poses = templates[0].size();

     for (size_t object = 0; object < templates.size(); ++object) {
         DBfeats.push_back(computeDescriptors(CNN, templates[object]));
         DBtest.push_back(computeDescriptors(CNN, test_set[object]));
     }

     // Create a k-NN matcher
     cv::Ptr<DescriptorMatcher> matcher;
     vector< vector<DMatch> > matches;
     matcher = DescriptorMatcher::create("BruteForce");
     matcher->add(DBfeats);
     int knn = 1;
     matcher->knnMatch(DBtest, matches, knn);

     vector<float> bins = {-1, 0, 10, 20, 40, 180};
     vector<float> histo(bins.size(), 0);
     float median_angle, mean_angle = 0;
     vector<float> diff_vector;

     for (size_t linearId = 0; linearId < (unsigned)DBtest.rows; ++linearId) {

         // - get the test set indices (1D -> 2D)
         int query_object = linearId / nr_test_poses;
         int query_pose = linearId % nr_test_poses;

         for (int nn = 0; nn < knn; nn++)
         {
             int knn_object =  matches[linearId][nn].trainIdx / nr_tmpl_poses;
             int knn_pose =  matches[linearId][nn].trainIdx % nr_tmpl_poses;
#if 0
             imshow("query",showRGBDPatch(test_set[query_object][query_pose].data,false));
             imshow("KNN",showRGBDPatch(templates[knn_object][knn_pose].data,false));
             waitKey();
#endif

             // If objects are different fill the first bin
             if (test_set[query_object][query_pose].label.at<float>(0,0) != templates[knn_object][knn_pose].label.at<float>(0,0))
             { histo[0]++; continue; }

             // Get the quaternions
             Quaternionf query_quat, knn_quat;
             for (int q = 0; q < 4; ++q) {
                 query_quat.coeffs()(q) = test_set[query_object][query_pose].label.at<float>(0,1+q);
                 knn_quat.coeffs()(q) = templates[knn_object][knn_pose].label.at<float>(0,1+q);
             }

             // Angular difference
             float diff;
             // If object is normal compare angular distance, else elevation level
             if(rotInv[test_set[query_object][query_pose].label.at<float>(0,0)] == 0) {
                 diff = query_quat.angularDistance(knn_quat)*180.f/M_PI;
             } else {
                 diff = abs(acos(query_quat.toRotationMatrix()(2,2)) - acos(knn_quat.toRotationMatrix()(2,2)))*180.f/M_PI;
             }
             mean_angle += diff;
             diff_vector.push_back(diff);

             // Exact match?
             Quaternionf tmpl_quat;
             bool exact_match = true;
             for (int tmpl_pose = 0; tmpl_pose < nr_tmpl_poses; ++tmpl_pose) {
                 for (int q = 0; q < 4; ++q)
                     tmpl_quat.coeffs()(q) = templates[knn_object][tmpl_pose].label.at<float>(0,1+q);
                 if (query_quat.angularDistance(tmpl_quat) < query_quat.angularDistance(knn_quat))
                     exact_match = false;
             }
             if (exact_match) histo[1]++;


             for (size_t b = 2; b < bins.size(); ++b) {
                 if (diff < bins[b])
                 {
                     histo[b]++;
                 }
             }
         }
     }

     // Get mean angular difference
     mean_angle /= DBtest.rows;
     // Get median angular difference
     sort(diff_vector.begin(), diff_vector.end());
     median_angle = diff_vector[(diff_vector.size()-1)/2];

     float total = histo.front() + histo.back();
     for (float &i : histo) i /= total;

     // Write stats to the file
     // - set the file name
     // -- initialize the parser
     boost::property_tree::ptree pt;
     boost::property_tree::ini_parser::read_ini(config, pt);

     // -- read learning parameters
     bool use_real = pt.get<bool>("input.use_real");
     bool random_background = pt.get<bool>("input.random_background");
     string rb = random_background ? "_rb" : "";
     string train_data = use_real ? "_real+synth" : "_synth";
     string net_name = pt.get<string>("train.net_name");
     vector<string> used_models = to_array<string>(pt.get<string>("input.used_models"));
     string eval_name = "eval_" + to_string(used_models.size())+ "_" + net_name + train_data + rb +".log";

     ofstream stat_file;
     stat_file.open(eval_name, std::ios_base::app);

     stat_file << "Iteration: " << iter << endl;
     for (size_t i = 0; i < histo.size(); ++i)
         stat_file << "<" << bins[i] << "\t";
     stat_file << "mean" << "\t" << "median" << endl;

     for (size_t i = 0; i < histo.size(); ++i)
         stat_file << std::setprecision(3) << histo[i]*100 << "\t";
     stat_file << mean_angle << "\t" << median_angle << endl;

     stat_file.close();
}
