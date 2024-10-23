
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected
        self.middle_count = 0


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #NUM_BINS = 3
        
        done = False

        gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        gray_frame = gray_frame[-100:, :]

        _, mask = cv2.threshold(gray_frame, 130, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        state = np.zeros(20)
        if contours: 
            cv2.drawContours(cv_image, max(contours, key=cv2.contourArea), -1, (0, 255, 0), 2)
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                #_, width = gray_frame.shape
                bin_width = 32
                bin_height = 10 #unhinged name 
                #print(cx, bin_width, cx//bin_width)
                state[cx//bin_width] = 1
                state[10 + cy//bin_height] = 1
                cv2.circle(cv_image, (cx, cy), radius=3, color=(255, 255, 255), thickness=-1)  # Red dot with radius 5
            else:
                print("weird error")
            self.timeout = 0
        else:
            #print("skipped")
            self.timeout+=1
            if self.timeout >= 30:
                done = True

        cv2.putText(cv_image, str(state), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 2)

        cv2.imshow("Raw Image", cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('Quit')

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def calculate_reward(self, state):
        '''
        Reward function based on how close the agent is to the ideal state [0,0,0,0,1,0,0,0,0]
        '''
        if np.argmax(state) in [3, 4, 5]:
            self.middle_count += 1  # Increment if in the middle
        else:
            self.middle_count = 0  # Reset if not in the middle

        # Base reward (higher if the line is closer to the center)
        ideal_position = 4  # Ideal position (center of the state array)
        current_position = np.argmax(state)  # Find the index where the '1' is located
        distance_from_ideal = abs(current_position - ideal_position)

        base_reward = max(0, 10 - 2 * distance_from_ideal)  # Reward decreases as distance increases
        
        # Apply a multiplier based on how long the line has stayed in the middle
        multiplier = 1 + (self.middle_count * 0.1)  # Example: 10% bonus for each consecutive step in the middle

        reward = base_reward * multiplier
        
        return reward

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.0
        elif action == 1:  # FORWARD LEFT
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 0.3
        elif action == 2:  # FORWARD RIGHT
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = -0.3
        elif action == 3: # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 4: #RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            reward = self.calculate_reward(state)
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
