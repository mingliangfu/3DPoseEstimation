
This project was created as a part of the "Real-Time 3D Object Classification and Pose Estimation Based on Deep Neural Networks" course. It provides the implementation of an object tracker based on an autoencoder network. You can find the description of the classes and their methods in the header files and more detailed comments in the source files. For additional information please see the presentation slides under misc folder.

##Description:
3D object classification and pose estimation are essential tasks in several developing scientific fields including robotics and augmented reality. However, there are still many issues that hinder the progress of these tasks, e.g. image noise, illumination changes, object occlusions, lack of powerful features, scarcity of reliable training data, scalability with respect to the number of classes, etc. As a part of this work, we investigate different approaches to tackle the problem of 3D pose estimation and object classification. In particular the triplet network proposed by [Wohlhart et al.](https://cvarlab.icg.tugraz.at/projects/3d_object_detection/) is taken as a baseline for the research. The aim of the work is then to study the scalability of the baseline and improve it in terms of accuracy, training and execution times by using various optimization techniques and knowledge gathered from studying other approaches solving the same problem.

##Installation instructions:

download a dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html, e.g. Dudek dataset: http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Dudek.zip.
add the autoencoderlib folder containing the autoencoder framework into the root folder.
change the path to the dataset in the main.cpp and set a ground truth rectangle.
