#include "../include/hdf5handler.h"


namespace sz {

hdf5Handler::hdf5Handler()
{

}

vector<Sample> hdf5Handler::read(string filename)
{
    vector<Sample> samples;
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet data_set = file.openDataSet("data");
        H5::DataSet label_set = file.openDataSet("label");

        H5::DataSpace data_space = data_set.getSpace();
        H5::DataSpace label_space = label_set.getSpace();

        vector<hsize_t> data_dims(data_space.getSimpleExtentNdims());
        vector<hsize_t> label_dims(label_space.getSimpleExtentNdims());
        data_space.getSimpleExtentDims(data_dims.data(), nullptr);
        label_space.getSimpleExtentDims(label_dims.data(), nullptr);

        assert(data_dims[0] == label_dims[0]); // Make sure that data count = label count

        // Specify size and shape of subset to read. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> slab_size = {1, data_dims[1], data_dims[2],data_dims[3]};
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        // Copy from Caffe layout back into OpenCV layout
        vector<float> temp(data_dims[1]*data_dims[2]*data_dims[3]);   // Memory for one patch
        samples.resize(data_dims[0]);
        for (uint i=0; i < samples.size(); ++i)
        {
            // Select correct patch as hyperslab inside the HDF5 file
            offset[0] = i;
            data_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            data_set.read(temp.data(), H5::PredType::NATIVE_FLOAT, memspace, data_space);

            samples[i].data = Mat(data_dims[2],data_dims[3],CV_32FC(data_dims[1]));
            for (hsize_t chan=0; chan < data_dims[1]; ++chan)
            {
                auto currChan = chan*data_dims[2]*data_dims[3];
                for (hsize_t y =0; y < data_dims[2]; ++y)
                    for (hsize_t x =0; x < data_dims[3]; ++x)
                        samples[i].data.ptr<float>(y)[x*data_dims[1] + chan] = temp[currChan + y*data_dims[3] + x];
            }
        }

        slab_size = {1, label_dims[1]};
        memspace = H5::DataSpace(slab_size.size(), slab_size.data());
        for (uint i=0; i < samples.size(); ++i)
        {
            // Select correct label as hyperslab inside the HDF5 file
            samples[i].label = Mat(1,label_dims[1],CV_32F);
            offset[0] = i;
            label_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            label_set.read(samples[i].label.data, H5::PredType::NATIVE_FLOAT, memspace, label_space);
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
    return samples;
}

void hdf5Handler::write(string filename, vector<Sample> &samples)
{
    if (samples.empty())
    {
        cerr << "writeHDF5: Nothing to write!" << endl;
        return;
    }

    try
    {
        // The HDF5 data layer in Caffe expects the following (everything float!):
        // Data must be Samples x Channels x Height x Width
        // Labels must be Samples x FeatureDim
        vector<hsize_t> data_dims = {samples.size(),(hsize_t) samples[0].data.channels(),(hsize_t) samples[0].data.rows,(hsize_t) samples[0].data.cols};

        // Specify size and shape of subset to write. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> slab_size = {1, data_dims[1], data_dims[2],data_dims[3]};

        H5::DataSpace data_space(data_dims.size(),data_dims.data());
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DataSet data_set = file.createDataSet("data",H5::PredType::NATIVE_FLOAT,data_space);

        // Fill temporary memory how Caffe needs it by filling the correct hyperslab per sample in the HDF5
        vector<float> temp(data_dims[1]*data_dims[2]*data_dims[3]);
        for(uint i=0; i < samples.size(); ++i)
        {
            for (hsize_t ch = 0; ch < data_dims[1]; ++ch)
                for (hsize_t y = 0; y < data_dims[2]; ++y)
                    for (hsize_t x = 0; x < data_dims[3]; ++x)
                        temp[ch*data_dims[2]*data_dims[3] + y*data_dims[3] + x] = samples[i].data.ptr<float>(y)[x*data_dims[1] + ch];

            offset[0] = i;
            data_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            data_set.write(temp.data(), H5::PredType::NATIVE_FLOAT, memspace, data_space);
        }

        vector<hsize_t> label_dims = {samples.size(), (hsize_t) samples[0].label.cols};
        H5::DataSpace label_space(label_dims.size(),label_dims.data());
        H5::DataSet label_set = file.createDataSet("label",H5::PredType::NATIVE_FLOAT,label_space);
        slab_size = {1, label_dims[1]};
        memspace = H5::DataSpace(slab_size.size(), slab_size.data());
        for(uint i=0; i < samples.size(); ++i)
        {
            offset[0] = i;
            label_space.selectHyperslab(H5S_SELECT_SET, slab_size.data(),offset.data());
            label_set.write(samples[i].label.data, H5::PredType::NATIVE_FLOAT, memspace, label_space);
        }

    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}



void hdf5Handler::writeTensorFlow(string filename, vector<Sample> &samples)
{
    if (samples.empty())
    {
        cerr << "writeHDF5: Nothing to write!" << endl;
        return;
    }
    try
    {
        detectionTUM::Sample &s = samples[0];
        vector<hsize_t> d_dims = {samples.size(),(hsize_t)s.data.cols,(hsize_t)s.data.rows,(hsize_t)s.data.channels()};
        vector<hsize_t> l_dims = {samples.size(),(hsize_t)s.label.cols,(hsize_t)s.label.rows,(hsize_t)s.label.channels()};

        // Specify size and shape of subset to write. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> d_slab_size = {1, d_dims[1], d_dims[2], d_dims[3]};
        vector<hsize_t> l_slab_size = {1, l_dims[1], l_dims[2], l_dims[3]};

        H5::DataSpace d_space(d_dims.size(),d_dims.data());
        H5::DataSpace d_mem(d_slab_size.size(), d_slab_size.data());
        H5::DataSpace l_space(l_dims.size(),l_dims.data());
        H5::DataSpace l_mem(l_slab_size.size(), l_slab_size.data());

        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DataSet d_set = file.createDataSet("data",H5::PredType::NATIVE_FLOAT,d_space);
        H5::DataSet l_set = file.createDataSet("label",H5::PredType::NATIVE_FLOAT,l_space);
        for(uint i=0; i < samples.size(); ++i)
        {
            offset[0] = i;
            d_space.selectHyperslab(H5S_SELECT_SET, d_slab_size.data(),offset.data());
            d_set.write(samples[i].data.data, H5::PredType::NATIVE_FLOAT, d_mem, d_space);
            l_space.selectHyperslab(H5S_SELECT_SET, l_slab_size.data(),offset.data());
            l_set.write(samples[i].label.data, H5::PredType::NATIVE_FLOAT, l_mem, l_space);
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}

vector<Sample> hdf5Handler::readTensorFlow(string filename)
{
    vector<Sample> samples;
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet d_set = file.openDataSet("data");
        H5::DataSet l_set = file.openDataSet("label");

        H5::DataSpace d_space = d_set.getSpace();
        H5::DataSpace l_space = l_set.getSpace();

        vector<hsize_t> d_dims(d_space.getSimpleExtentNdims());
        vector<hsize_t> l_dims(l_space.getSimpleExtentNdims());
        d_space.getSimpleExtentDims(d_dims.data(), nullptr);
        l_space.getSimpleExtentDims(l_dims.data(), nullptr);

        assert(d_dims[0] == l_dims[0]); // Make sure that data count = label count

        // Specify size and shape of subset to read. Additionally needed memspace
        vector<hsize_t> offset = {0, 0, 0, 0};
        vector<hsize_t> d_slab_size = {1, d_dims[1], d_dims[2],d_dims[3]};
        vector<hsize_t> l_slab_size = {1, l_dims[1], l_dims[2],l_dims[3]};

        H5::DataSpace d_mem(d_slab_size.size(), d_slab_size.data());
        H5::DataSpace l_mem(l_slab_size.size(), l_slab_size.data());

        samples.resize(d_dims[0]);
        if (count>0) samples.resize(count);
        for(uint i=0; i < samples.size(); ++i)
        {
            offset[0] = i;
            samples[i].data =  Mat(d_dims[1],d_dims[2],CV_32FC(d_dims[3]));
            samples[i].label = Mat(l_dims[1],l_dims[2],CV_32FC(l_dims[3]));
            d_space.selectHyperslab(H5S_SELECT_SET, d_slab_size.data(),offset.data());
            d_set.read(samples[i].data.data, H5::PredType::NATIVE_FLOAT, d_mem, d_space);
            l_space.selectHyperslab(H5S_SELECT_SET, l_slab_size.data(),offset.data());
            l_set.read(samples[i].label.data, H5::PredType::NATIVE_FLOAT, l_mem, l_space);
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
    return samples;
}




Isometry3f hdf5Handler::readBBPose(string filename)
{
   Isometry3f poseMat;
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet pose = file.openDataSet("H_table_from_reference_camera");
        H5::DataSet offset = file.openDataSet("board_frame_offset");

        H5::DataSpace pose_space = pose.getSpace();
        H5::DataSpace offset_space = offset.getSpace();

        vector<hsize_t> pose_dims(pose_space.getSimpleExtentNdims());
        vector<hsize_t> offset_dims(offset_space.getSimpleExtentNdims());
        pose_space.getSimpleExtentDims(pose_dims.data(), nullptr);
        offset_space.getSimpleExtentDims(offset_dims.data(), nullptr);

        vector<hsize_t> slab_size = {pose_dims[0],pose_dims[1]};
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        float newbuffer[pose_dims[0]][pose_dims[1]]; //static array.
        pose.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, pose_space);

        for (size_t x = 0; x < pose_dims[0]; ++x) {
            for (size_t y = 0; y < pose_dims[1]; ++y) {
                poseMat.matrix()(x,y) = newbuffer[x][y];
            }
        }
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
    return poseMat;
}

Mat hdf5Handler::readBBDepth(string filename)
{
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet depth = file.openDataSet("depth");
        H5::DataSpace depth_space = depth.getSpace();
        vector<hsize_t> depth_dims(depth_space.getSimpleExtentNdims());
        depth_space.getSimpleExtentDims(depth_dims.data(), nullptr);

        vector<hsize_t> slab_size = {depth_dims[0],depth_dims[1]};
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        float newbuffer[depth_dims[0]][depth_dims[1]]; //static array.
        depth.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, depth_space);
        Mat depthImage = Mat(depth_dims[0],depth_dims[1], CV_32F);

        for (size_t x = 0; x < depth_dims[0]; ++x) {
            for (size_t y = 0; y < depth_dims[1]; ++y) {
                depthImage.at<float>(x,y) = newbuffer[x][y];
            }
        }
        // Convert to meters
        depthImage.convertTo(depthImage,CV_32F,0.0001f);

        return depthImage;
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}

Matrix3f hdf5Handler::readBBIntristicMats(string filename)
{
    vector<Matrix3f,Eigen::aligned_allocator<Matrix3f>> cams(5, Matrix3f());
    Matrix3f argcam = Matrix3f::Zero();
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet np1 = file.openDataSet("NP1_depth_K");
        H5::DataSet np2 = file.openDataSet("NP2_depth_K");
        H5::DataSet np3 = file.openDataSet("NP3_depth_K");
        H5::DataSet np4 = file.openDataSet("NP4_depth_K");
        H5::DataSet np5 = file.openDataSet("NP5_depth_K");

        H5::DataSpace np_space = np1.getSpace();
        vector<hsize_t> np_dims(np_space.getSimpleExtentNdims());
        np_space.getSimpleExtentDims(np_dims.data(), nullptr);

        vector<hsize_t> slab_size = {np_dims[0],np_dims[1]};
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        float newbuffer[np_dims[0]][np_dims[1]];

        np1.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                cams[0].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np2.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                cams[1].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np3.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                cams[2].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np4.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                cams[3].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np5.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                cams[4].matrix()(x,y) = newbuffer[x][y];
            }
        }
        // Average all the intristic cameras
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                for (size_t sum = 0; sum < cams.size(); ++sum) {
                    argcam(x,y) += cams[sum].matrix()(x,y);
                }
                argcam(x,y) /= cams.size();
            }
        }
        return cams[0];
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}

vector<Isometry3f,Eigen::aligned_allocator<Isometry3f>> hdf5Handler::readBBTrans(string filename)
{
    vector<Isometry3f,Eigen::aligned_allocator<Isometry3f>> trans(5, Isometry3f());
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet np1 = file.openDataSet("H_NP1_from_NP5");
        H5::DataSet np2 = file.openDataSet("H_NP2_from_NP5");
        H5::DataSet np3 = file.openDataSet("H_NP3_from_NP5");
        H5::DataSet np4 = file.openDataSet("H_NP4_from_NP5");
        H5::DataSet np5 = file.openDataSet("H_NP5_from_NP5");

        H5::DataSpace np_space = np1.getSpace();
        vector<hsize_t> np_dims(np_space.getSimpleExtentNdims());
        np_space.getSimpleExtentDims(np_dims.data(), nullptr);

        vector<hsize_t> slab_size = {np_dims[0],np_dims[1]};
        H5::DataSpace memspace(slab_size.size(), slab_size.data());

        float newbuffer[np_dims[0]][np_dims[1]]; //static array.

        np1.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                trans[0].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np2.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                trans[1].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np3.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                trans[2].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np4.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                trans[3].matrix()(x,y) = newbuffer[x][y];
            }
        }
        np5.read(newbuffer, H5::PredType::NATIVE_FLOAT, memspace, np_space);
        for (size_t x = 0; x < np_dims[0]; ++x) {
            for (size_t y = 0; y < np_dims[1]; ++y) {
                trans[4].matrix()(x,y) = newbuffer[x][y];
            }
        }
        return trans;
    }
    catch(H5::Exception error)
    {
        error.printError();
        assert(0);
    }
}

}
