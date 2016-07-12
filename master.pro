TARGET = cnn_test
QT += opengl
CONFIG   += console
CONFIG -= app_bundle


LIBS +=  -L/usr/local/lib -L/opt/tum/external/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
INCLUDEPATH += /usr/local/include /usr/include/eigen3 /usr/include/hdf5/serial/ $$PWD/include/ /opt/tum/external/include /usr/local/cuda-8.0/include/

QMAKE_CXXFLAGS += -std=c++11 -march=native -O3
LIBPATH += /home/zsn/Documents/caffe/distribute/lib
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_viz -lhdf5_cpp -lhdf5 -lcaffe -lprotobuf -lglog -lopencv_features2d -lopencv_photo

# Linux only
unix:!macx {
    LIBS +=  -lboost_filesystem -lboost_system -lboost_program_options
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

SOURCES += main.cpp \
    src/sphere.cpp  \
    src/painter.cpp \
    src/model.cpp  \
    src/datasetmanager.cpp \
    src/networksolver.cpp \
    src/hdf5handler.cpp \
    src/utilities.cpp \
    src/networkevaluator.cpp



HEADERS += include/sphere.h \
    include/painter.h \
    include/model.h \
    include/datasetmanager.h \
    include/datatypes.h \
    include/networksolver.h \
    include/hdf5handler.h \
    include/utilities.h



