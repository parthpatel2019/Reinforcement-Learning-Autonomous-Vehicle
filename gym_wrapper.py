import numpy as np
import time
import gym_donkeycar
import gym
from way_points import WayPoints
from Reward import getReward


class DonkeySimEnv:
    #############################################################################
    # This is the configuration that will initialize the car into DonkeyCar Gym
    #############################################################################
    def __init__(self):
        conf = {
            "exe_path": f"./donkey_sim.exe", # Not Necessay To Change, Can Ignore
            "host": "127.0.0.1", # Host IP (Either Local or Remote Server)
            "port": 9091, # Port of Connection. Match with Server Connection
            "body_style": "f1", # car01 | f1
            "body_rgb": (0, 0, 0), # (Red, Green, Blue)
            "car_name": "RL_Model", # Car Name
            "font_size": 100, # Size of Name Above Car
            "racer_name": "MyRLModel", # Name To Appear In Scoreboard
            "country": "USA", 
            "bio": "RL Model",
            "max_cte": 5, # Distance From The Center of The Track
            "fov": 80,
            "fish_eye_x": 0.0,
            "fish_eye_y": 0.0,
            "img_w": 224, # Image Depth
            "img_h": 224, # Image Height
            "img_d": 3, # Number of Channels
            "img_enc": 'JPG', # Type of Image Encoding
            "offset_x": 0.0,  # Sides
            "offset_y": 2.0,  # Height
            "offset_z": 0.0,  # Forward
            "rot_x": 20.0,  # Tilt
        }
        self.env = gym.make('donkey-circuit-launch-track-v0', conf=conf) # Populate the car into the simulator
        self.current_time = None # Amount of time the car has been on the track before the reward functions outputs done 
        observationReceived, reward, done, info = self.env.step([0, 0]) # Initial step command that sends command to the simulator
        self.imageObservation = observationReceived # Image returned from the Car's Camera
        self.TimeForScoreEnabled = 1 # TimerForScoreEnabled=1 indicates the car has not started moving so timer is not enabled
        self.waypoints = WayPoints(textFile='circuit_points.txt').returnWayPoints() # Get waypoints for the specific track (Used for reward function)
        self.default_throttle = 0.5 # Default throttle value

    ###################################################################################
    # Send the command to the vehicle to proceed in a certain direction
    # @param: action - Type: Integer - Steering direction that corresponds to the steering value in the steering dictionary
    ###################################################################################
    def step(self, action):
        # If restart == 1, this means this is the first move the car is making from the starting line 
        # so it will start a timer to see how long it lasts on the track
        if self.TimeForScoreEnabled == 1:
            self.current_time = time.time()
            self.TimeForScoreEnabled = 0

        steering_dictionary = { # The current steering values it can take
            0: -0.4,
            1: 0,
            2: 0.4
        }

        send_move = [steering_dictionary[np.argmax(action)], self.default_throttle] # [Steering, Throttle] that is sent to the simulator
        observationReceived, _, done, info = self.env.step(send_move)
        self.imageObservation = observationReceived
        reward_value = getReward(observationReceived=observationReceived, done=done, info=info, waypoints=self.waypoints)
        print(reward_value)
        return observationReceived, reward_value, done, info, (time.time() - self.current_time) # How long the car has stayed on the track for the current iteration

    ###################################################################################
    # Obtain the latest image that car receives
    ###################################################################################
    def get_state(self):
        return self.imageObservation

    ###################################################################################
    # Reset the car
    ###################################################################################
    def reset(self):
        obs = self.env.reset()
        for point in self.waypoints:
          point.hit = False
        self.TimeForScoreEnabled = 1

    ###################################################################################
    # Take the car out of the simulator
    ###################################################################################
    def close(self):
        self.current_time = None
        self.env.close()
