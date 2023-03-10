import numpy as np
import os
import math
import cv2

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import rospkg
import threading

import kinetic_controller


HOST_NAME = os.environ["VEHICLE_NAME"]
PUBLISH_IMAGE = True
PUBLISH_IMAGE_TYPE = 'red'
PROCESSING_RATE = 20


class LaneFollowingNode:
    def __init__(self):
        # Initialize an instance of Renderer giving the model in input.
        self.count = 0
        self.image_lock = threading.Lock()
        self.sub = rospy.Subscriber(f'/{HOST_NAME}/camera_node/image/compressed', CompressedImage, self.callback)

        if PUBLISH_IMAGE:
            self.pub = rospy.Publisher(f'/{HOST_NAME}/lane_following/compressed', CompressedImage, queue_size=10)
        self.image = None
        self.seq = 0

        ANGLE_MULT = 0.
        POSITION_MULT = 1.
        self.controller = kinetic_controller.KineticController(
            (ANGLE_MULT * .3, ANGLE_MULT * .002, ANGLE_MULT * -3), 
            (POSITION_MULT * .005, POSITION_MULT * .00005, POSITION_MULT * -.0))
        
        self.max_speed = 0.55  # top speed when driving in a single lane
        self.speed = self.max_speed  # current speed

        self.turn_flag = False
        self.stop_timer_default = PROCESSING_RATE * .25  # time before stopping after seeing a red line
        self.stop_timer = self.stop_timer_default  # current timer, maxed out at self.stop_timer_default
        self.turn_detection = [0., 0., 0.]  # detecting if the left, forward and right direction of an intersection has a road to turn to

        self.continue_run = True
        self.last_angle_error = 0.
        self.last_position_error = 0.
        def general_callback(msg):
            if msg.data == 'stop':
                self.continue_run = False
        rospy.Subscriber('/general', String, general_callback)

    def callback(self, msg):
        # how to decode compressed image
        # reference: http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
        self.count += 1
        if self.count % 2 == 0:
            compressed_image = np.frombuffer(msg.data, np.uint8)
            im = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
            self.image_lock.acquire()
            self.image = im
            self.image_lock.release()
    
    def general_callback(self, msg):
        strs = msg.data.split()
        if len(strs) == 4:
            cp, ci, cd = float(strs[1]), float(strs[2]), float(strs[3])
            if strs[0] == 'position':
                print(f'setting position coefficients to {cp} {ci} {cd}')
                self.controller.position_coeffs = (cp, ci, cd)
            elif strs[0] == 'angle':
                print(f'setting angle coefficients to {cp} {ci} {cd}')
                self.controller.angle_coeffs = (cp, ci, cd)
            else:
                print(f'coefficient type {strs[0]} not recognized!')

    def run(self):
        rate = rospy.Rate(PROCESSING_RATE)  # in Hz
        for i in range(10):
            self.controller.drive(0, 0)
            rate.sleep()

        while not rospy.is_shutdown():
            if not self.continue_run:
                self.controller.drive(0, 0)
                break

            self.image_lock.acquire()
            im = self.image
            self.image_lock.release()
            if im is not None:
                self.update_controller(im)
                self.stopline_processing(im)
                self.controller.update()
            rate.sleep()
    
    def update_controller(self, im):
        publish_flag = PUBLISH_IMAGE and PUBLISH_IMAGE_TYPE == 'yellow'
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        lower_range = np.array([22,100,150])
        upper_range = np.array([30,255,255])

        yellow_mask = cv2.inRange(hsv, lower_range, upper_range)
        yellow_mask[:260, 330:] = 0
        img_dilation = cv2.dilate(yellow_mask, np.ones((35, 35), np.uint8), iterations=1)

        contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # pick the largest contour
        largest_area = 0
        largest_idx = -1
        refx, refy = im.shape[1] * 0.5, 130.5
        for i in range(len(contours)):
            ctn = contours[i]
            xmin, ymin, width, height = cv2.boundingRect(ctn)
            midx, midy = xmin + .5 * width, ymin + .5 * height
            if midy < 190 or midx + midy < 290:  # crop top half
                continue
            area = cv2.contourArea(ctn)
            if area > largest_area:
                largest_area = area
                largest_idx = i

        vx, vy, cosref, sinref = 1, 0, 1, 0
        position_ref = 0
        contour_y = 0
        contour_x = 0
        if largest_idx != -1:
            largest_ctn = contours[largest_idx]
            if publish_flag:
                im = cv2.drawContours(im, contours, largest_idx, (0,255,0), 3)
            [vx,vy,x,y] = cv2.fitLine(largest_ctn, cv2.DIST_L2,0,0.01,0.01)
            vx, vy = vx[0], vy[0]
            if vx + vy > 0:
                vx, vy = -vx, -vy
            angle = math.atan2(vy, vx)

            xmin, ymin, width, height = cv2.boundingRect(largest_ctn)
            contour_x, contour_y = xmin + width * 0.5, ymin + height * 0.5
            ref_angle = math.atan2(refy - contour_y, refx - contour_x)     
            cosref, sinref = math.cos(ref_angle), math.sin(ref_angle)
            angle_error = (ref_angle - angle + math.pi) % (2 * math.pi) - math.pi
            if contour_y >= 420 or (contour_x - refx) ** 2 + (contour_y - refy) ** 2 < 155 ** 2:
                angle_error = 0.

            down_right_pt_x = 320. + 120. * (self.stop_timer / self.stop_timer_default)
            position_line_ref = np.cross(
                np.array((im.shape[1] * 0.5, 130.5, 1.)), 
                np.array((70., down_right_pt_x, 1.)))
            position_line_ref /= position_line_ref[0]
            position_ref = -position_line_ref[2].item() - contour_y * position_line_ref[1].item()
            position_error = position_ref - contour_x

            if contour_x <= 10 or contour_y >= 420 or (contour_x - refx) ** 2 + (contour_y - refy) ** 2 < 155 ** 2:
                angle_error = self.last_angle_error
            
            self.last_angle_error = angle_error
            self.last_position_error = position_error
        else:
            angle_error = self.last_angle_error
            position_error = self.last_position_error
        
        position_error = max(position_error, -280.)
        self.controller.update_error(angle_error, position_error)
        adjust = self.controller.get_adjustment()

        adjust = max(min(adjust, .9), -.9)
        left_speed = self.speed * (1 - adjust)
        right_speed = self.speed * (1 + adjust)
        
        if self.controller.actionQueueIsEmpty():
            self.controller.driveForTime(left_speed, right_speed, 1)

        if publish_flag:
            ARROW_LENGTH = 50
            if largest_idx !=-1:
                if angle_error != 0:
                    cv2.arrowedLine(im,
                        (int(contour_x), int(contour_y)), 
                        (int(contour_x + vx * ARROW_LENGTH), int(contour_y + vy * ARROW_LENGTH)), 
                        (255, 0, 0), 3)
                    cv2.arrowedLine(im,
                        (int(contour_x), int(contour_y)), 
                        (int(contour_x + cosref * ARROW_LENGTH), int(contour_y + sinref * ARROW_LENGTH)), 
                        (0, 255, 0), 3)
                cv2.arrowedLine(im,
                    (int(contour_x + position_error), int(contour_y)), 
                    (int(contour_x + position_error), int(contour_y - ARROW_LENGTH)), 
                    (0, 0, 255), 3)
            msg = CompressedImage()
            msg.header.seq = self.seq
            msg.header.stamp = rospy.Time.now()
            msg.format = 'jpeg'
            ret, buffer = cv2.imencode('.jpg', im)
            if not ret:
                print('failed to encode image!')
            else:
                msg.data = np.array(buffer).tostring()
                self.pub.publish(msg)
                self.seq += 1
    
    def stopline_processing(self, im):
        publish_flag = PUBLISH_IMAGE and PUBLISH_IMAGE_TYPE == 'red'
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        lower_range = np.array([0,70,120])
        upper_range = np.array([5,180,255])

        red_mask = cv2.inRange(hsv, lower_range, upper_range)
        img_dilation = cv2.dilate(red_mask, np.ones((10, 10), np.uint8), iterations=1)

        contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        left, right = self.controller.getCurrentSpeeds()
        vehicle_is_waiting = abs(left) + abs(right) < .1
        # pick the largest contour
        largest_area = 0
        largest_idx = -1
        for i in range(len(contours)):
            ctn = contours[i]
            area = cv2.contourArea(ctn)

            xmin, ymin, width, height = cv2.boundingRect(ctn)
            xmax = xmin + width
            midx, midy = xmin + .5 * width, ymin + .5 * height

            # detect which way we can turn to
            if vehicle_is_waiting and (area > 500 and im.shape[0] * 0.55 > midy > im.shape[0] * 0.33):
                if len(self.controller.actions_queue) > 2:  # forward-facing
                    if midx < im.shape[1] * 0.45:
                        print(f'case1 {midx}, {midy}')
                        self.turn_detection[1] += .5
                    elif midx < im.shape[1] * 0.9:
                        print(f'case2 {midx}, {midy}')
                        self.turn_detection[2] += 1
                else:  # left-facing
                    if midx < im.shape[1] * .5:
                        print(f'case3 {midx}, {midy}')
                        self.turn_detection[0] += 1
                    else:
                        print(f'case4 {midx}, {midy}')
                        self.turn_detection[1] += .5

            if area > largest_area and area > 1000 and xmax > im.shape[1] * .5 and xmin < im.shape[1] * .5:
                largest_area = area
                largest_idx = i

        contour_y = 0
        if largest_idx != -1:
            largest_ctn = contours[largest_idx]

            if publish_flag:
                im = cv2.drawContours(im, contours, largest_idx, (0,255,0), 3)

            xmin, ymin, width, height = cv2.boundingRect(largest_ctn)
            contour_y = ymin + height * 0.5

        if self.turn_flag:
            if self.controller.actionQueueIsEmpty():
                # make a turn
                min_idx = 0
                for i in range(1, len(self.turn_detection)):
                    if self.turn_detection[i] < self.turn_detection[0]:
                        min_idx = i
                turn_idx = (min_idx + 1) % 3

                self.speed = self.max_speed
                if turn_idx == 0:
                    self.controller.driveForTime(.6 * self.speed, 1.4 * self.speed, PROCESSING_RATE * .75)
                elif turn_idx == 1:
                    self.controller.driveForTime(1.2 * self.speed, .8 * self.speed, PROCESSING_RATE * .75)
                elif turn_idx == 2:
                    self.controller.driveForTime(1.8 * self.speed, .2 * self.speed, PROCESSING_RATE * .75)

                # reset the detection list since we are out of the intersection after the turn
                for i in range(len(self.turn_detection)):
                    self.turn_detection[i] = 0
                self.turn_flag = False

        print(contour_y)
        if contour_y > 430 or (contour_y > 420 and self.stop_timer < self.stop_timer_default):
            print("stopping")
            self.speed = 0
            self.stop_timer -= 1
        if self.stop_timer <= 0:  # prepare to go into intersection
            self.stop_timer = self.stop_timer_default + 30
            # for now, always turn right
            self.turn_flag = True
            self.controller.driveForTime(-1., 1., PROCESSING_RATE * .25)
            self.controller.driveForTime(0., 0., PROCESSING_RATE * .25)
            self.controller.driveForTime(1., -1., PROCESSING_RATE * .15)
        else:  # not approaching stop line
            self.speed = self.max_speed
            if self.stop_timer > self.stop_timer_default:
                self.stop_timer = max(self.stop_timer - 1, self.stop_timer_default)
            else:
                self.stop_timer = min(self.stop_timer + 1, self.stop_timer_default)
                

        if publish_flag:
            contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im = cv2.drawContours(im, contours, -1, (0,255,0), 3)

            msg = CompressedImage()
            msg.header.seq = self.seq
            msg.header.stamp = rospy.Time.now()
            msg.format = 'jpeg'
            ret, buffer = cv2.imencode('.jpg', im)
            if not ret:
                print('failed to encode image!')
            else:
                msg.data = np.array(buffer).tostring()
                self.pub.publish(msg)
                self.seq += 1


def entry():
    ar_node = LaneFollowingNode()
    ar_node.run()
