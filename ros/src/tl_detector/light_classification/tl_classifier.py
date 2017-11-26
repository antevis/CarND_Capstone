from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np

from scipy.misc import imresize as resize


# Off-the-shelf SSD Mobilenet COCO trained model from TensorFlow detection model zoo
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
PATH_TO_CKPT = 'model/ssd_mobilenet_coco.pb'

class TLClassifier(object):
    def __init__(self):

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                print("model loaded")

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)

    def crop(self, image_np, bbox=[.2,.4,.3,.6]):
        """
        Crops the given image according to values given in bbox
        :param image_np:
        :param bbox: crop values (relative)
        :return: image_np cropped to the bounding box
        """
        im_height, im_width = image_np.shape[0], image_np.shape[1]

        y1 = int(im_height * bbox[0])
        x1 = int(im_width * bbox[1])
        y2 = int(im_height * bbox[2])
        x2 = int(im_width * bbox[3])

        return image_np[y1:y2, x1:x2, :]


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # reducing image size by a factor of 4 to speed-up the detection. Still accurate.
        image = resize(image, (150,200))

        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        np_bboxes = np.squeeze(boxes)
        np_scores = np.squeeze(scores)
        np_classes = np.squeeze(classes).astype(np.int32)

        traffic_light_boxes_mask = [(tl and conf) for tl, conf in zip(np_classes == 10, np_scores > .5)]

        # Direct Boolean array indexing in Python 2 doesn't work the same as in Python 3
        tl_boxes = [bbox for bbox, m in zip(np_bboxes, traffic_light_boxes_mask) if m]

        # tl_boxes now contains only bounding boxes that the model is confident are the traffic lights

        for box in tl_boxes:

            # Cropping to the bounding box
            tl_image_np = self.crop(image_np=image, bbox=box)

            #cropping to the roi withing the bounding box
            roi = self.crop(image_np=tl_image_np)

            # average red channel intensity
            red_avg = np.average(roi[:, :, 0])

            if red_avg > 200:
                # print("   RED detected")
                return TrafficLight.RED

        # print("no RED detected")
        return TrafficLight.UNKNOWN
        # return TrafficLight.RED
