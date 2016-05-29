
TARGET = cnn_test
QT += opengl
CONFIG   += console
CONFIG -= app_bundle


LIBS +=  -L/usr/local/lib
INCLUDEPATH += /usr/local/include /usr/include/eigen3 /usr/include/hdf5/serial/


QMAKE_CXXFLAGS += -std=c++11 -march=native -O3
LIBPATH += /home/zsn/Documents/caffe/distribute/lib
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_viz -lhdf5_cpp -lhdf5 -lcaffe -lprotobuf -lglog -lopencv_features2d

unix:!macx {
    LIBS +=  -lboost_filesystem -lboost_system

}

macx: {
    QMAKE_CXXFLAGS += -stdlib=libc++
    INCLUDEPATH += /opt/local/include/eigen3/ /opt/local/include/
    INCLUDEPATH += /Developer/NVIDIA/CUDA-7.5/include
    LIBS += -L/opt/local/lib/ -lboost_filesystem-mt -lboost_system-mt
}



#QMAKE_CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
#QMAKE_CFLAGS += -fsanitize=address -fno-omit-frame-pointer
#QMAKE_LFLAGS += -fsanitize=address

SOURCES += main.cpp  sphere.cpp  painter.cpp  model.cpp sphere.h painter.h model.h



