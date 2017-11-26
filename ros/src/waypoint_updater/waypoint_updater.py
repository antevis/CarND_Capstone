#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

# used to obtain proximal waypoints. Borrowed from Aaron Brown's implementation
from scipy.spatial import KDTree

import math

import copy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number

# Introduced this class to abstract waypoint management
class WayPointsManager():
    def __init__(self):

        self.wps_tree = None
        self.wp_container = None

    def waypoints_cb(self, waypoints_msg):

        if not self.wp_container:
            self.wp_container = waypoints_msg
            wps_data = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.wp_container.waypoints]
            self.wps_tree = KDTree(wps_data)

    def get_proximal_wp(self, pose):
        return self.wps_tree.query([pose.position.x, pose.position.y], 1)[1]

    # borrowed the idea of deepcopy from 'Kung-fu Panda automotive' team:
    # https://github.com/kung-fu-panda-automotive/carla-driver/blob/5151ffdcdef5faa3947e43a5f1ceb16a75f9ddfe/ros/src/waypoint_updater/waypoint_helper.py#L112
    def look_ahead_wps(self, start_idx, wp_count):
        terminal_idx = max(0,min(len(self.wp_container.waypoints), start_idx + wp_count))

        lane = Lane()
        lane.header = self.wp_container.header
        lane.waypoints = copy.deepcopy(self.wp_container.waypoints[start_idx+1:terminal_idx])

        return lane


class WaypointUpdater(object):
    def __init__(self):

        self.wp_manager = WayPointsManager()

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        ## Published once
        rospy.Subscriber('/base_waypoints', Lane, self.wp_manager.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Introduced to avoid computing it again in tl_detector
        self.curr_wp_idx_pub = rospy.Publisher('vehicle_wp_idx', Int32, queue_size=1)

        self.currPos = None
        self.tl_ws_idx = -1
        self.proximal_wp_idx = None
        self.loop()


    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.currPos and self.wp_manager.wp_container:
                self.proximal_wp_idx = self.wp_manager.get_proximal_wp(self.currPos.pose)
                self.publish_lookahead(self.proximal_wp_idx)
                self.curr_wp_idx_pub.publish(self.proximal_wp_idx)
            rate.sleep()


    def pose_cb(self, msg):
        self.currPos = msg

    def traffic_cb(self, msg):
        self.tl_ws_idx = msg.data


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def publish_lookahead(self, closest_waypoint):

        if not closest_waypoint or not self.wp_manager.wp_container or closest_waypoint < 0 or closest_waypoint >= len(self.wp_manager.wp_container.waypoints):

            return

        if self.tl_ws_idx != -1:

            curr_velocity = self.get_waypoint_velocity(self.wp_manager.wp_container.waypoints[closest_waypoint])

            step_count = self.tl_ws_idx - closest_waypoint
            velocity_decrement = curr_velocity / step_count if step_count > 0 else curr_velocity

            lane = self.wp_manager.look_ahead_wps(start_idx=closest_waypoint, wp_count=step_count)

            for i in range(len(lane.waypoints)):
                curr_velocity -= velocity_decrement

                self.set_waypoint_velocity(lane.waypoints, i, curr_velocity)
        else:

            lane = self.wp_manager.look_ahead_wps(start_idx=closest_waypoint, wp_count=LOOKAHEAD_WPS)

        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
