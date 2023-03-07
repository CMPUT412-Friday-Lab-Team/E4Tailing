
import rospy
from duckietown_msgs.srv import ChangePattern
from std_msgs.msg import String


class LEDClient:
    def __init__(self, hostname):
        rospy.wait_for_service(f'/{hostname}/led_emitter_node/set_pattern')
        self.change_pattern_srv = rospy.ServiceProxy(f'/{hostname}/led_emitter_node/set_pattern', ChangePattern)
        self.cur_pattern_str = None
    
    def change_pattern(self, pattern_str):
        if pattern_str == None or self.cur_pattern_str == pattern_str:
            return
        self.cur_pattern_str = pattern_str

        try:
            msg = String()
            msg.data = pattern_str
            self.change_pattern_srv(msg)
            print(f'changing pattern to {pattern_str}')
        except rospy.ServiceException as e:
            print('Service request failed')
            print(e)