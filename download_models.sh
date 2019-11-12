OUTPUT_DIR=$(pwd)/face_detection_model
wget https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb -O "${OUTPUT_DIR}/opencv_face_detector_uint8.pb"
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt -O "${OUTPUT_DIR}/opencv_face_detector.pbtxt"