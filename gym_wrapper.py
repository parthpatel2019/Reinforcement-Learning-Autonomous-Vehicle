import numpy as np
import time
import gym_donkeycar
from gyminterface import GymInterface
import gym
from way_points import WayPoints
from Reward import getReward
import random


class DonkeySimEnv:
    #############################################################################
    # This is the configuration that will initialize the car into DonkeyCar Gym
    #############################################################################
    def __init__(self):
        GYM_DICT = {
            'car': {
                'car_name': 'RLModel',
                'font_size': 50,
                'racer_name': 'Triton AI',
                'bio': 'Something',
                'country': 'US',
                'body_style': 'f1',
                'body_rgb': [0, 0, 0],
                'guid': 'some_random_string'},

            # Which is the default connection profile? "local" or "remote"?
            'default_connection': 'local',
            # default_connection: 'remote'

            'local_connection': {
                # roboracingleague_1 | generated_track | generated_road | warehouse | sparkfun_avc | waveshare
                'scene_name': 'donkey-circuit-launch-track-v0',
                # Use "127.0.0.1" for simulator running on local host.
                'host': '127.0.0.1',
                'port': 9091,
                'artificial_latency': 0},  # Ping the remote simulator whose latency you would like to match with, and put the ping in millisecond here.

            'remote_connection': {
                'scene_name': 'generated_track',
                # Use the actual host name for remote simulator.
                'host': '127.0.0.1',
                'port': 9091,
                'artificial_latency': 0},  # Besides the ping to the remote simulator, how many MORE delay would you like to add?

            'lidar': {
                "degPerSweepInc": "2",
                "degAngDown": "0",
                "degAngDelta": "0",
                "num_sweeps_levels": "1",
                "maxRange": "50.0",
                "noise": "0",
                "offset_x": "0",
                "offset_y": "0",
                "offset_z": "0",
                "rot_x": "0",
                'deg_inc': 2, 
                'max_range': 50.0,
                'enabled': False},  # Max range of the lidar laser
            
            'camera': {
                "fov": 0,
                "fish_eye_x": 0.0,
                "fish_eye_y": 0.0,
                "img_w": 160,
                "img_h": 120,
                "img_d": 3,
                "img_enc": 'JPG',
                "offset_x": 0,
                "offset_y": 0,
                "offset_z": 0,
                "rot_x": 0,
                # "rot_y": 180,
            },
        }
        self.gym_env = GymInterface(gym_config=GYM_DICT)
        self.current_time = None # Amount of time the car has been on the track before the reward functions outputs done 
        observationReceived, tele, lidar, x, y, z, speed, cte = self.gym_env.step(0, 0, 0, True)  # Initial step command that sends command to the simulator
        time.sleep(2)
        observationReceived, tele, lidar, x, y, z, speed, cte = self.gym_env.step(0, 0, 0, True)
        self.imageObservation = observationReceived # Image returned from the Car's Camera
        self.TimeForScoreEnabled = 1 # TimerForScoreEnabled=1 indicates the car has not started moving so timer is not enabled
        self.waypoints = WayPoints(textFile='circuit_points.txt').returnWayPoints() # Get waypoints for the specific track (Used for reward function)
        self.default_throttle = 0.6 # Default throttle value
        self.MAX_CTE = 5
        self.pos_x = 0
        self.pos_z = 0

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

        send_move = [steering_dictionary[np.argmax(action)], self.default_throttle, 0, False] # [Steering, Throttle] that is sent to the simulator
        observationReceived, tele, lidar, x, y, z, speed, cte = self.gym_env.step(steering_dictionary[np.argmax(action)], self.default_throttle, 0, False)
        self.imageObservation = observationReceived
        done = False
        if ((abs(cte) > self.MAX_CTE and abs(cte) < 8) or abs(self.pos_x - x) < 1 and abs(self.pos_z - z) < 1 and speed < 1 and (time.time() - self.current_time) > 2):
            done = True
        self.pos_x = x
        self.pos_z = z
        reward_value = getReward(observationReceived=observationReceived, tele=tele, done=done, waypoints=self.waypoints)
        # How long the car has stayed on the track for the current iteration
        return observationReceived, reward_value, done, (time.time() - self.current_time)

    ###################################################################################
    # Obtain the latest image that car receives
    ###################################################################################
    def get_state(self):
        return self.imageObservation

    ###################################################################################
    # Reset the car
    ###################################################################################
    def reset(self):
        obs = self.gym_env.reset_car()
        for point in self.waypoints:
          point.hit = False
        self.TimeForScoreEnabled = 1

    ###################################################################################
    # Take the car out of the simulator
    ###################################################################################
    def close(self):
        self.current_time = None
        self.gym_env.onShutdown()

    def teleport(self):
        selectedPoint = random.randint(0, len(self.waypoints)-2)
        self.gym_env.teleport_car(
            self.waypoints[selectedPoint].x, 0.56, self.waypoints[selectedPoint].z)
