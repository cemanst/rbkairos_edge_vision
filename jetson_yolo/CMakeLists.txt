cmake_minimum_required(VERSION 3.0.2)
project(jetson_yolo)

## Koristimo C++14 standard (podrzan od tvog g++)
add_compile_options(-std=c++14)

## ROS komponente
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
)

## Sistemske zavisnosti (CUDA i OpenCV)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

## Putanje do zaglavlja (Headers)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /usr/local/cuda-10.2/include
  /usr/include/aarch64-linux-gnu
)
link_directories(
  /usr/local/cuda-10.2/lib64
  ${catkin_LIBRARY_DIRS}
)
catkin_package()
## Definisanje izvrsnog fajla
add_executable(yolo_node src/yolo_node.cpp)

## Linkovanje binarnih biblioteka
target_link_libraries(yolo_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  nvinfer
  cudart
)

