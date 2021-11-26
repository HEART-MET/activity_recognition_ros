cmake_minimum_required(VERSION 2.8.3)
project(activity_recognition_ros)

find_package(catkin REQUIRED)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    sensor_msgs
    cv_bridge
    metrics_refbox_msgs
  DEPENDS
    OpenCV
)
