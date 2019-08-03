TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp
INCLUDEPATH += /home/julien/opencv/build/include
LIBS += -L/home/julien/opencv/build/lib
LIBS += -lopencv_core -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_video -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts
LIBS+= /usr/lib/x86_64-linux-gnu/librealsense2.so.2.24
# see https://github.com/IntelRealSense/librealsense/issues/4351   if still not working, might need to switch back to 2.23