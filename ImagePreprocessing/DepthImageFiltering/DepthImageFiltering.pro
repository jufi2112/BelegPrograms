TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

INCLUDEPATH += /home/julien/opencv/build/include
LIBS += -L/home/julien/opencv/build/lib
LIBS += -lopencv_core -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_video -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts
LIBS += -lstdc++fs
