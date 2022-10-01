
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from collections import deque

class Listener:

    def __init__(self, topic_name, topic_type, maxlen=5, verbose_log=False, callback=None):
        self.topic_name, self.topic_type = topic_name, topic_type
        if callback is not None:
            self.callback = callback
        else:
            self.callback = self.__callback
        rospy.Subscriber(self.topic_name, self.topic_type, self.callback)
        # self.data_stack = deque([], maxlen=maxlen)
        self.last_dat = None
        self.verbose_log = verbose_log

    def __callback(self, data):
        # self.data_stack.append(data)
        self.last_dat = data
        if self.verbose_log:
            rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)

    def get_data(self, timeout=5):
        return rospy.wait_for_message(self.topic_name, self.topic_type,
                                      timeout=timeout)

    ##
    # @brief spin() simply keeps python from exiting until this node is stopped
    def spin(self):
        rospy.spin()