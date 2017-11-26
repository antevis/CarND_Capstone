#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from math import cos, sin, atan2
import datetime

STATE_COUNT_THRESHOLD = 3
MKZ_LEN = 4.93  # Length of the vehicle in meters

# used to obtain proximal waypoints. Borrowed from Aaron Brown's implementation
from scipy.spatial import KDTree


class WayPointsManager():

    def __init__(self, stop_line_coords):
        self.wps_tree = None
        self.wp_container = None
        self.wps_data = None
        self.sl_data = stop_line_coords

        self.stopLine_tree = KDTree(self.sl_data)

        self.sl_wp_indices = {}

    def waypoints_cb(self, waypoints_msg):

        if not self.wp_container:
            self.wp_container = waypoints_msg

            self.wps_data = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.wp_container.waypoints]
            self.wps_tree = KDTree(self.wps_data)

            # Lookup table of global waypoints indices for each stop line
            self.sl_wp_indices = {i: self.get_proximal(self.sl_data[i], self.wps_tree)[1]
                                  for i in range(len(self.sl_data))}

            """
            Adjustment to the vehicle's body length.
            Might've been complicated further by introducing vehicle's heading.

            May seem computationally expensive but performed once."""
            for sl_idx, wp_idx in self.sl_wp_indices.items():

                wp_coord = self.wps_data[wp_idx]

                yaw = 0

                if wp_idx > 0:
                    wp_coord_prev = self.wps_data[wp_idx-1]

                    yaw = atan2((wp_coord[1]-wp_coord_prev[1]), (wp_coord[0]-wp_coord_prev[0]))

                localX = wp_coord[0] * cos(-yaw) - wp_coord[1] * sin(-yaw)
                localY = wp_coord[0] * sin(-yaw) + wp_coord[1] * cos(-yaw)

                localX -= MKZ_LEN / 2

                globalX = localX * cos(yaw) - localY * sin(yaw)
                globalY = localX * sin(yaw) + localY * cos(yaw)

                self.sl_data[sl_idx] = [globalX, globalY]

            # Re-assemble the stop line - waypoint lookup table again according to adjusted stop line waypoints.
            # The whole of the above might be refactored and optimized, but I am not in the mood.
            self.sl_wp_indices = {i: self.get_proximal(self.sl_data[i], self.wps_tree)[1]
                                  for i in range(len(self.sl_data))}


    def get_proximal_wp(self, pose):

        position = [pose.position.x, pose.position.y]
        return self.get_proximal(coords=position, tree=self.wps_tree)


    def get_proximal_sl(self, pose):

        position = [pose.position.x, pose.position.y]
        return self.get_proximal(coords=position, tree=self.stopLine_tree)


    def get_proximal(self, coords, tree):

        return tree.query(x=coords, k=1)



class TLDetector(object):
    def __init__(self):

        rospy.init_node('tl_detector')

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        stop_line_positions = self.config['stop_line_positions']

        self.wp_manager = WayPointsManager(stop_line_positions)

        self.pose = None
        self.camera_image = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.wp_manager.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        # Topic published by waypoint updater. To avoid redundant computations locally, obtain it by subscription.
        rospy.Subscriber('/vehicle_wp_idx', Int32, self.ego_curr_wp_idx)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.state_count = 0
        self.tls = None
        self.light_wp_idx = -1
        self.ego_wp_idx = -1

        self.loop()

    def traffic_cb(self, msg):
        self.tls = msg.lights

    def ego_curr_wp_idx(self, msg):
        self.ego_wp_idx = msg.data

    def loop(self):
        rate = rospy.Rate(10)  # 50Hz. Maybe change this
        while not rospy.is_shutdown():

            if self.state_count >= STATE_COUNT_THRESHOLD:

                self.upcoming_red_light_pub.publish(Int32(self.light_wp_idx))

            rate.sleep()


    def pose_cb(self, msg):
        self.pose = msg


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.camera_image = msg
        self.light_wp_idx, state = self.process_traffic_lights()

        if state != self.state:

            self.state = state
            self.state_count = 0

        else:
            self.state_count += 1


    def get_light_state(self):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.camera_image:

            # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            # "passthrough" encoding returns image as numpy ndarray (which is what I want as an input to the TF model)
            np_image = self.bridge.imgmsg_to_cv2(img_msg=self.camera_image,
                                                 desired_encoding="passthrough")
            #Get classification
            return self.light_classifier.get_classification(np_image)
        else:
            return self.state

    def process_traffic_lights(self):

        if (self.pose):
            sl_distance, tl_idx = self.wp_manager.get_proximal_sl(self.pose.pose)
            sl_wp_idx = self.wp_manager.sl_wp_indices[tl_idx]

            wp_gap = sl_wp_idx - self.ego_wp_idx

            # waypoint gap introduced to allow vehicle leave the intersection and not stop again at the same traffic line
            tl_ahead = wp_gap > -5 and (sl_distance < 50)

            if tl_ahead:

                state = self.get_light_state()  # Detection using TensorFlow model.
                # state = self.tls[tl_idx].state  # pseudo "Detection" from /vehicle/traffic_lights topic

                if state == TrafficLight.RED:
                    light_wp_idx = sl_wp_idx
                else:
                    light_wp_idx = -1

                return light_wp_idx, state

            else:
                return -1, TrafficLight.UNKNOWN

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

