
TARGET = cnn_test
QT += opengl
CONFIG   += console
CONFIG -= app_bundle


LIBS +=  -L/usr/local/lib -lboost_filesystem
INCLUDEPATH += /usr/local/include /usr/include/eigen3 /usr/include/hdf5/serial/


QMAKE_CXXFLAGS += -std=c++11 -march=native -O3
LIBPATH += /home/zsn/Documents/caffe/distribute/lib
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_viz -lboost_system -lhdf5_cpp -lhdf5 -lcaffe -lprotobuf -lglog -lopencv_features2d


#QMAKE_CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
#QMAKE_CFLAGS += -fsanitize=address -fno-omit-frame-pointer
#QMAKE_LFLAGS += -fsanitize=address

SOURCES += example.cpp  sphere.cpp  painter.cpp  model.cpp sphere.h painter.h model.h



