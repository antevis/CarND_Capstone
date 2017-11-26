[//]: # (Image References)

[elephant]: ./imgs/detection_elephant.png "elephant"
[red]: ./imgs/detection_red.png "red"
[green]: ./imgs/detection_green.png "green"
[crop0]: ./imgs/crop0.png "crop"
[crop1]: ./imgs/crop1.png "crop"
[crop2]: ./imgs/crop2.png "crop"
[crop3]: ./imgs/crop3.png "crop"
[crop4]: ./imgs/crop4.png "crop"
[crop5]: ./imgs/crop5.png "crop"
[crop6]: ./imgs/crop6.png "crop"
[crop7]: ./imgs/crop7.png "crop"
[crop8]: ./imgs/crop8.png "crop"
[crop9]: ./imgs/crop9.png "crop"
[crop10]: ./imgs/crop10.png "crop"
[crop11]: ./imgs/crop11.png "crop"
[roi0]: ./imgs/roi0.png "crop"
[roi1]: ./imgs/roi1.png "crop"
[roi2]: ./imgs/roi2.png "crop"
[roi3]: ./imgs/roi3.png "crop"
[roi4]: ./imgs/roi4.png "crop"
[roi5]: ./imgs/roi5.png "crop"
[roi6]: ./imgs/roi6.png "crop"
[roi7]: ./imgs/roi7.png "crop"
[roi8]: ./imgs/roi8.png "crop"
[roi9]: ./imgs/roi9.png "crop"
[roi10]: ./imgs/roi10.png "crop"
[roi11]: ./imgs/roi11.png "crop"
[roi12]: ./imgs/roi12.png "crop"
[screen]: ./imgs/screen.png "not moving while detected"

# CarND Capstone Project
---
### Self-Driving Car Engineer Nanodegree Program

Traffic light classification utilizes [off-the-shelf trained SSD Mobilenet COCO model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) available at
[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#tensorflow-detection-model-zoo). The model detects **80 classes** of objects, **including traffic lights**, and for each input image it produces 4 outputs (3 arrays and 1 scalar value):
* bounding boxes
* scores
* classes
* number of detections (that is, length of the above)

![alt text][elephant] ![alt text][red] ![alt text][green]

Bounding boxes with traffic lights look like this:

![alt text][crop0] ![alt text][crop1] ![alt text][crop2] ![alt text][crop3] ![alt text][crop4] ![alt text][crop5]
![alt text][crop6] ![alt text][crop7] ![alt text][crop8] ![alt text][crop9] ![alt text][crop10] ![alt text][crop11]

As **the model doesn't distinguish traffic lights states**, additional classification required to discern **`REDs`**.
To do that, I **further crop** the detected bounding box **to the center of the top quadrant** and perform primitive **average red channel intensity evaluation**:

![alt text][roi0]![alt text][roi1]![alt text][roi2]![alt text][roi3]![alt text][roi4]![alt text][roi5]![alt text][roi6]
![alt text][roi7]![alt text][roi8]![alt text][roi9]![alt text][roi10]![alt text][roi11]![alt text][roi12]

It can be seen that minimum average red channel intensity among the examined detections is **`205.3095`** (in a range of **`0÷255`**). So I deem **`200`** to be a **`threshold`** value for the detection to be classified as **`RED`**.

Despite the model is relatively lightweight and aimed for efficient inference on mobile devices, my current local hardware setup **([Mid-2014 3.0GHz dual-core Intel Core i7 16GB laptop](https://support.apple.com/kb/SP703?locale=en_US&viewlocale=en_US) + Lubunu VM with ROS Kinetic)** is apparently unable to cope with the task. The vehicle successfully starts, stops on a **`red`** and proceeds on a **`green`** in case of **pseudo-"detections"** consumed from the **`/vehicle/traffic_lights`** topic, but fails to do so based on the data from the  **`tl_classifier`**, while the latter provides a stream of correct predictions:

![alt text][screen]

The virtual machine meets the recommended configuration specs: **`2 CPUs + 4GB RAM`**

As the project due is in 1 day, I submit "as is." Even if it wouldn't pass, hope for a meaningful feedback from the
reviewer.


