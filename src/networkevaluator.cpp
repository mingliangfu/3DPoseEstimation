#include "include/networkevaluator.h"

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
                int currImg = j*img_size;
                for (int ch=0; ch < channels ; ++ch)
                    for (int y = 0; y < targetSize; ++y)
                        for (int x = 0; x < targetSize; ++x)
                            data[currImg + slice*ch + y*targetSize + x] = patch.ptr<float>(y)[x*channels + ch];
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

void networkEvaluator::visualizeManifold(caffe::Net<float> &CNN, const vector<vector<Sample>> &templates, int iter)
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

void networkEvaluator::computeHistogram(caffe::Net<float> &CNN, const vector<vector<Sample>> &templates, const vector<vector<Sample>> &training_set, const vector<vector<Sample>> &test_set, vector<int> rotInv)
{
    // Get the test data
    Mat DBfeats, DBtest;
    vector<Sample> test, tmpl;

    for (size_t object = 0; object < templates.size(); ++object) {
        DBfeats.push_back(computeDescriptors(CNN, templates[object]));
        DBtest.push_back(computeDescriptors(CNN, test_set[object]));
        for (Sample s : test_set[object]) test.push_back(s);
        for (Sample s : templates[object]) tmpl.push_back(s);
    }

    // Create a k-NN matcher
    cv::Ptr<DescriptorMatcher> matcher;
    matcher = DescriptorMatcher::create("BruteForce");
    matcher->add(DBfeats);

    vector< vector<DMatch> > matches;
    int knn = 1;
    matcher->knnMatch(DBtest, matches, knn);

    vector<float> bins = {-1, 5, 20, 40, 180};
    vector<float> histo(bins.size(), 0);

    for (int i=0; i < DBtest.rows; ++i)
        for (DMatch &m : matches[i])
        {
            Mat query = showRGBDPatch(test[m.queryIdx].data, false);
            Mat nn = showRGBDPatch(tmpl[m.trainIdx].data, false);
            // cerr << DBtest.row(m.queryIdx) << endl;
            // cerr << DBfeats.row(m.trainIdx) << endl;
            imshow("query",query); imshow("nn", nn); waitKey();

            // If objects are different fill the first bin
            if (test[m.queryIdx].label.at<float>(0,0) != tmpl[m.trainIdx].label.at<float>(0,0))
            {
                histo[0]++;
                continue;
            }

            // Get the quaternions
            Quaternionf sampleQuat, kNNQuat;
            for (int q = 0; q < 4; ++q)
            {
                sampleQuat.coeffs()(q) = test[m.queryIdx].label.at<float>(0,1+q);
                kNNQuat.coeffs()(q) = tmpl[m.trainIdx].label.at<float>(0,1+q);
            }

            // Angular difference
            float diff;
            // If object is normal compare angular distance, else elevation level
            if(rotInv[test[m.queryIdx].label.at<float>(0,0)] == 0) {
                diff = sampleQuat.angularDistance(kNNQuat)*180.f/M_PI;
            } else {
                diff = abs(acos(sampleQuat.toRotationMatrix()(2,2)) - acos(kNNQuat.toRotationMatrix()(2,2)))*180.f/M_PI;
            }

            for (size_t b = 1; b < bins.size(); ++b)
                if (diff < bins[b])
                {
                    histo[b]++;
                    break;
                }
        }

    float total=0;
    for (float &i : histo) total += i;
    for (float &i : histo) i /= total;
    for (size_t i=0; i < histo.size(); ++i)
        clog << i << " " << histo[i] << endl;

}
