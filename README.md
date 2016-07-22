#Real-Time 3D Object Classification and Pose Estimation Based on Deep Neural Networks
This project was created as a part of the "Real-Time 3D Object Classification and Pose Estimation Based on Deep Neural Networks" master's thesis project.

##Description:
3D object classification and pose estimation are essential tasks in several developing scientific fields including robotics and augmented reality. However, there are still many issues that hinder the progress of these tasks, e.g. image noise, illumination changes, object occlusions, lack of powerful features, scarcity of reliable training data, scalability with respect to the number of classes, etc. As a part of this work, we investigate different approaches to tackle the problem of 3D pose estimation and object classification. In particular the triplet network proposed by [Wohlhart et al.](https://cvarlab.icg.tugraz.at/projects/3d_object_detection/) is taken as a baseline for the research. The aims of the work are to study the scalability of the baseline and its generalization possibilities, to adapt it to be able to learn on big datasets efficiently and improve it in terms of accuracy and execution time by using various optimization techniques and knowledge gathered from studying other approaches solving the same problem.

##Installation instructions:

**Required packages:**
* [CUDA](https://developer.nvidia.com/cuda-downloads)
  * [CUDA 7.5 hack - gcc 5](https://gist.github.com/wangruohui/679b05fcd1466bb0937f)
* Eigen3: *sudo apt install libeigen3-dev*
* VTK6: *sudo apt install libvtk6-dev*
* [OpenCV 3.1](http://opencv.org/downloads.html)
* HDF5: *sudo apt install libhdf5-serial-dev*
  * HDF5 fix:
	*find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;*
	*cd /usr/lib/x86_64-linux-gnu*
	*sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so*
	*sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so*
* Caffe: Download [the latest release](https://github.com/BVLC/caffe) and add [the triplet layer](https://github.com/BVLC/caffe/pull/2603)
  * Before installation take care of prerequisites (Boost, BLAS, etc) as mentioned [here](http://caffe.berkeleyvision.org/installation.html).
* OpenGL Utility Toolkit: *sudo apt-get install freeglut3-dev*


##Used datasets:
* [LineMOD](https://cvarlab.icg.tugraz.at/projects/3d_object_detection/)
* [BigBIRD](http://rll.berkeley.edu/bigbird/)
* [Washington](http://rgbd-dataset.cs.washington.edu/dataset/)
