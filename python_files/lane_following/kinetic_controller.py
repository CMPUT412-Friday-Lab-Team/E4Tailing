import rospy
from duckietown_msgs.msg import WheelsCmdStamped
import os
import math


HOST_NAME = os.environ["VEHICLE_NAME"]


P_DECAY_FACTOR = 0.5
I_DECAY_FACTOR = 0.02
D_DECAY_FACTOR = 0.05


class KineticController:
    def __init__(self, angle_coeffs, position_coeffs):
        """
        cp, ci and cd are all need to be smaller than 0
        cp < 0
        ci < 0
        cd > 0
        """
        self.pub = rospy.Publisher(f'/{HOST_NAME}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=10)
        def myhook():
            self.stop()
        rospy.on_shutdown(myhook)

        self.angle_coeffs = angle_coeffs
        self.position_coeffs = position_coeffs

        self.angle_error_i = 0  # integrated
        self.angle_error_p = 0
        self.angle_error = 0
        self.angle_error_d = 0
        self.position_error_i = 0
        self.position_error_p = 0
        self.position_error = 0
        self.position_error_d = 0
        self.adjustment = 0

        self.turn_timer = 0
        self.actions_queue = []
    
    def stop(self):
        """
        stay still to reduce the momentum of the car to zero after carrying out some movement
        """
        self.drive(0, 0)

    def drive(self, left_speed, right_speed):
        msg = WheelsCmdStamped()
        msg.vel_left = left_speed
        msg.vel_right = right_speed
        self.pub.publish(msg)
    
    def actionQueueIsEmpty(self):
        return len(self.actions_queue) == 0

    def getCurrentState(self):
        if len(self.actions_queue) == 0:
            return None
        else:
            state = self.actions_queue[0][3]
            return state

    def update(self, is_car_too_close):
        if len(self.actions_queue) == 0:
            print('0 action queue')
            self.drive(0., 0.)
        else:
            left_speed, right_speed, time, state = self.actions_queue[0]
            print(left_speed, right_speed, time, state)
            if (left_speed == 0 and right_speed == 0) or not is_car_too_close:
                self.drive(left_speed, right_speed)
                if time <= 1:
                    self.actions_queue.pop(0)
                else:
                    self.actions_queue[0] = (left_speed, right_speed, time - 1, state)
            else:
                self.drive(0., 0.)

    def driveForTime(self, left_speed, right_speed, ntime_step, state):
        self.actions_queue.append((left_speed, right_speed, ntime_step, state))
    
    def update_error(self, angle_error, position_error):
        self.angle_error_i = (self.angle_error_i + angle_error) * (1 - I_DECAY_FACTOR)
        self.position_error_i = (self.position_error_i + position_error) * (1 - I_DECAY_FACTOR)

        self.angle_error_d = self.angle_error_d * (1 - D_DECAY_FACTOR) + (angle_error - self.angle_error) * D_DECAY_FACTOR
        self.position_error_d = self.position_error_d * (1 - D_DECAY_FACTOR) + (position_error - self.position_error) * D_DECAY_FACTOR

        self.angle_error = angle_error
        self.position_error = position_error

        self.angle_error_p = self.angle_error_p * (1 - P_DECAY_FACTOR) + angle_error * P_DECAY_FACTOR
        self.position_error_p = self.position_error_p * (1 - P_DECAY_FACTOR) + position_error * P_DECAY_FACTOR

        ap, ai, ad = self.angle_coeffs
        pp, pi, pd = self.position_coeffs

        angle_adjustment = self.angle_error_i * ai + self.angle_error_p * ap + self.angle_error_d * ad
        position_adjustment = self.position_error_i * pi + self.position_error_p * pp + self.position_error_d * pd
        self.adjustment = angle_adjustment + position_adjustment

    def get_adjustment(self):
        return self.adjustment

