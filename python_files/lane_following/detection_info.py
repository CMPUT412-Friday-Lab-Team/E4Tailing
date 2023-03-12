"""
This file defines a class that manages the detection information for the duckiebot in front of the current bot
"""


import os
import rospy
from std_msgs.msg import Float32
from duckietown_msgs.msg import BoolStamped, VehicleCorners
import threading


HOST_NAME = os.environ["VEHICLE_NAME"]
SAFE_DRIVING_DISTANCE = 0.30
SAFE_TURN_DISTANCE = SAFE_DRIVING_DISTANCE


class DetectionManager:
    def __init__(self):
        self.sub_duckie_distance = rospy.Subscriber(f'/{HOST_NAME}/duckiebot_distance_node/distance', Float32, self.duckie_distance_callback, queue_size=1)
        self.sub_duckie_center = rospy.Subscriber(f'/{HOST_NAME}/duckiebot_detection_node/centers', VehicleCorners, self.duckie_center_callback, queue_size=1)
        self.sub_duckie_detection = rospy.Subscriber(f'/{HOST_NAME}/duckiebot_detection_node/detection', BoolStamped, self.duckie_detection_callback, queue_size=1)

        self.duckie_center = (0., 0.)
        self.duckie_distance = 0.
        self.duckie_detected = False

        self.lock = threading.Lock()
    
    def isCarTooClose(self):
        return self.duckie_distance < SAFE_DRIVING_DISTANCE and self.duckie_detected

    def isDetected(self):
        return self.duckie_detected

    def isSafeToTurn(self):
        return (not self.duckie_detected) or self.getDistance() > SAFE_TURN_DISTANCE

    def getCenter(self):
        return self.duckie_center

    def getDistance(self):
        return self.duckie_distance
    
    def duckie_distance_callback(self, msg):
        print('DISTANCE CALLBACK!')
        self.lock.acquire()
        self.duckie_distance = msg.data
        self.lock.release()
            
    def duckie_center_callback(self, msg):
        print('CENTER CALLBACK!')
        if not msg.detection: 
            self.lock.acquire()
            self.duckie_detected = False
            self.lock.release()
        else:
            corners_list = msg.corners
            sumx, sumy = .0, .0
            NUM_CORNERS = 21
            center = None
            if len(corners_list) > NUM_CORNERS * .5:  # half of the dots are visible
                for i in range(len(corners_list)):
                    corner = corners_list[i]
                    sumx += corner.x
                    sumy += corner.y
                center = (sumx / NUM_CORNERS, sumy / NUM_CORNERS)
            
            self.lock.acquire()
            if center is not None:
                self.duckie_center = center
            self.duckie_detected = True
            self.lock.release()
    
    def duckie_detection_callback(self, msg):
        self.lock.acquire()
        self.duckie_detected = msg.data
        self.lock.release()