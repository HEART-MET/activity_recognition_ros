This repository contains a ROS node which uses a trained activity recognition model to recognize activities from a video source (as received by a subscriber to a `sensor_msgs/Image` topic).

The code accompanies [this](https://www.youtube.com/watch?v=8NDX74oG3ZU) video, and the code for training the I3D model can be found [here](https://github.com/HEART-MET/pytorch-i3d).

The checkpoint for the model fine-tuned on the [HEART-MET Activity Recognition](https://competitions.codalab.org/competitions/30423) validation dataset can be found in the `config` directory.


## Compile

The package depends on the [metrics_refbox_msgs](https://github.com/HEART-MET/metrics_refbox_msgs), so clone that repository into your catkin workspace before you compile.

Compile with:
```
catkin build
```
or
```
catkin_make
```



## Launch
Before launching, modify the topics and parameters in the launch file as required. In particular, change the `input_rgb_image` topic to the one from your camera.

```
roslaunch activity_recognition_ros recognize_activity.launch
```

## Test

Send a start command to the node
```
rostopic pub /recognize_activity/command metrics_refbox_msgs/Command "task: 3
command: 1
task_config: ''
uid: ''" -1
```

If you are testing with a ROS bagfile, play the bagfile now. If you are testing with a live camera, the node will already start publishing recognized activities.

You can view the debug image with:
```
rosrun image_view image_view image:=/recognize_activity/debug_image
```

The final output of the top 5 activities can be seen on the `/metrics_refbox_client/activity_recognition_result` topic.
```
rostopic echo /metrics_refbox_client/activity_recognition_result
```
