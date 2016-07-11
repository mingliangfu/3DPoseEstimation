#include "hdf5handler.h"

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
