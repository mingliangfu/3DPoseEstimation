TARGET = cnn_test
QT += opengl
CONFIG   += console
CONFIG -= app_bundle

<<<<<<< HEAD
LIBS +=  -L/usr/local/lib
INCLUDEPATH += /usr/local/include /usr/include/eigen3 /usr/include/hdf5/serial/
=======

LIBS +=  -L/usr/local/lib -L/opt/tum/external/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
INCLUDEPATH += /usr/local/include /usr/include/eigen3 /usr/include/hdf5/serial/ $$PWD/include/ /opt/tum/external/include
>>>>>>> 26e3cc95c3c655b97760561ff73933d332d5692f

QMAKE_CXXFLAGS += -std=c++11 -march=native -O3
LIBPATH += /home/zsn/Documents/caffe/distribute/lib
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_viz -lhdf5_cpp -lhdf5 -lcaffe -lprotobuf -lglog -lopencv_features2d

# Linux only
unix:!macx {
<<<<<<< HEAD
    LIBS +=  -lboost_filesystem -lboost_system
=======
    LIBS +=  -lboost_filesystem -lboost_system -lboost_program_options

>>>>>>> 26e3cc95c3c655b97760561ff73933d332d5692f
}

# Mac only
macx: {
    QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.11
    INCLUDEPATH += /opt/local/include/eigen3/ /opt/local/include/
    INCLUDEPATH += /Developer/NVIDIA/CUDA-7.5/include
    LIBS += -L/opt/local/lib/ -lboost_filesystem-mt -lboost_system-mt
}

#QMAKE_CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
#QMAKE_CFLAGS += -fsanitize=address -fno-omit-frame-pointer
#QMAKE_LFLAGS += -fsanitize=address

SOURCES += main.cpp  src/sphere.cpp  src/painter.cpp  src/model.cpp  \
    src/datasetgenerator.cpp \
    src/networksolver.cpp \
    src/hdf5handler.cpp

HEADERS += include/sphere.h include/painter.h include/model.h include/datasetgenerator.h \
    include/datatypes.h \
    include/networksolver.h \
    include/hdf5handler.h



