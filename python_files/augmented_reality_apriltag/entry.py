#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2

import rospy
import yaml
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from dt_apriltags import Detector
import rospkg
# import led_client

import tf2_ros
import geometry_msgs.msg
import tags
from std_msgs.msg import String, Header


HOST_NAME = os.environ["VEHICLE_NAME"]
IGNORE_DISTANCE = 1.0
MIN_CHANGE_LED_INTERVAL = 5
PUBLISH_AR_IMAGES = True


"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode:

    def __init__(self):
        rospack = rospkg.RosPack()
        self.sub = rospy.Subscriber(f'/{HOST_NAME}/camera_node/image/compressed', CompressedImage, self.callback)
        self.pose_pub = rospy.Publisher(f'/{HOST_NAME}/april_tag_node/pos_estimate', geometry_msgs.msg.TransformStamped, queue_size=10)
        self.image = None
        self.seq = 0
        self.intrinsic = self.readYamlFile(rospack.get_path('augmented_reality_apriltag') + '/src/camera_intrinsic.yaml')
        self.detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        # self.led_client = led_client.LEDClient(HOST_NAME)
        self.target_led_color = 'WHITE'
        self.last_detection_counter = 0
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

        self.continue_run = True
        def general_callback(msg):
            if msg.data == 'stop':
                self.continue_run = False
        rospy.Subscriber('/general', String, general_callback)

        if PUBLISH_AR_IMAGES:
            self.pub = rospy.Publisher(f'/{HOST_NAME}/april_tag_node/compressed', CompressedImage, queue_size=10)
            # self.info_pub = rospy.Publisher(f'/{HOST_NAME}/april_tag_node/camera_info', CameraInfo, queue_size=10)
            self.seq = 0

    
    def callback(self, msg):
        # how to decode compressed image
        # reference: http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
        compressed_image = np.frombuffer(msg.data, np.uint8)
        im = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
        self.image = im
    
    def run(self):
        rate = rospy.Rate(5)  # in Hz

        for i in range(3):
            tags.broadcast_all_tags()
            rate.sleep()

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'world'
        t.child_frame_id = f'{HOST_NAME}/footprint'
        t.transform.translation = geometry_msgs.msg.Vector3(.6, .3, 0.)
        t.transform.rotation = geometry_msgs.msg.Quaternion(0., 0., 0., 1.)
        self.pose_pub.publish(t)

        while not rospy.is_shutdown() and self.continue_run:
            if self.image is not None:

                camera_matrix = np.array(self.intrinsic["camera_matrix"]["data"]).reshape(3,3)
                camera_proj_mat = np.concatenate((camera_matrix, np.zeros((3, 1), dtype=np.float32)), axis=1)
                distort_coeff = np.array(self.intrinsic["distortion_coefficients"]["data"]).reshape(5,1)
                fx = camera_matrix[0][0].item()
                fy = camera_matrix[1][1].item()
                cx = camera_matrix[0][2].item()
                cy = camera_matrix[1][2].item()
                tag_size = 0.065  # in meters

                width = self.image.shape[1]
                height = self.image.shape[0]

                newmatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distort_coeff, (width,height), 1, (width,height))
                undistort_im = cv2.undistort(self.image, camera_matrix, distort_coeff, None, newmatrix)
                input_image = cv2.cvtColor(undistort_im, cv2.COLOR_BGR2GRAY)
                detected_tags = self.detector.detect(input_image, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size)

                dst = undistort_im
                for det in detected_tags:
                    id = det.tag_id
                    
                    ihom_pose = det.pose_t
                    if np.linalg.norm(ihom_pose) > IGNORE_DISTANCE:
                        continue

                    hom_pose = np.zeros((4, 1), dtype=np.float32)
                    hom_pose[0:3, :] = ihom_pose
                    hom_proj_pose_t = camera_proj_mat @ hom_pose
                    ihom_proj_pose_t = hom_proj_pose_t[0:2, :] / hom_proj_pose_t[2, 0]
                    proj_x, proj_y = int(ihom_proj_pose_t[0, 0].item()), int(ihom_proj_pose_t[1, 0].item())

                    if PUBLISH_AR_IMAGES:
                        cv2.drawMarker(dst, (proj_x, proj_y), (0, 0, 255))

                        cv2.polylines(dst, np.array([det.corners], dtype=np.int32), True, (255, 0, 0))
                        ihom_center = np.sum(np.array(det.corners, dtype=np.float32), axis=0) / 4
                        centerx, centery = int(ihom_center[0].item()), int(ihom_center[1].item())
                        cv2.drawMarker(dst, (centerx, centery), (0, 255, 0))
                    
                    name = None
                    if id in (162, 169):  # stop sign
                        name = "Stop sign"
                        target_led_color = 'RED'
                        
                    elif id in (58, 133, 62, 153):  # T-intersection
                        name = "T-intersection"
                        target_led_color = 'BLUE'
                    elif id in (93, 94, 200, 201):  # UofA tag
                        name = "UofA tag"
                        target_led_color = 'GREEN'
                    if name is not None:
                        if PUBLISH_AR_IMAGES:
                            dst = cv2.putText(dst, name, (centerx,centery), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 1, cv2.LINE_AA)
                        tags.broadcast_tag_estimate(det.pose_t, det.pose_R, f'{HOST_NAME}/april_est{id}', None, False)
                        # tags.broadcast_robot_pos_from_estimate(det.pose_t, det.pose_R, id, self.tfBuffer, self.pose_pub, True)
                        
                    if name != None and self.last_detection_counter > MIN_CHANGE_LED_INTERVAL:
                        self.target_led_color = target_led_color
                        self.last_detection_counter = 0
                
                if self.last_detection_counter > MIN_CHANGE_LED_INTERVAL:
                    self.target_led_color = 'WHITE'
                    self.last_detection_counter = 0

                # self.led_client.change_pattern(self.target_led_color)
                self.last_detection_counter += 1

                if PUBLISH_AR_IMAGES:
                    x,y,w,h = roi 
                    dst = dst[y:y+h, x:x+w]

                    msg = CompressedImage()
                    msg.header.seq = self.seq
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = f'{HOST_NAME}/camera_optical_frame'
                    msg.format = 'jpeg'
                    ret, buffer = cv2.imencode('.jpg', dst)
                    if not ret:
                        print('failed to encode image!')
                    else:
                        msg.data = np.array(buffer).tostring()
                        self.pub.publish(msg)
                        self.seq += 1

                        # publish camera info as well
                        # self.publishCameraInfo()

            rate.sleep()

    # def publishCameraInfo(self):
    #     # reference from https://gist.github.com/rossbar/ebb282c3b73c41c1404123de6cea4771
    #     info = CameraInfo()
    #     info.width = self.intrinsic['image_width']
    #     info.height = self.intrinsic['image_height']
    #     info.K = self.intrinsic['camera_matrix']['data']
    #     info.D = self.intrinsic['distortion_coefficients']['data']
    #     info.R = self.intrinsic['rectification_matrix']['data']
    #     info.P = self.intrinsic['projection_matrix']['data']
    #     info.distortion_model = self.intrinsic['distortion_model']
    #     self.info_pub.publish(info)

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def onShutdown(self):
        super(ARNode, self).onShutdown()


def entry():
    tags.broadcast_all_tags()
    ar_node = ARNode()
    ar_node.run()
