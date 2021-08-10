import os
import time
from pygame.locals import *
import pygame, time
import gyminterface

steer = 0.0
throttle = 0.0
brake = 0.0

track_name = 'donkey-circuit-launch-track-v0'

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
        'scene_name': track_name,
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
        "offset_y": 2.0,
        "offset_z": 0,
        "rot_x": 20.0,
        # "rot_y": 180,
    },
}

gym_env = GymInterface(gym_config=GYM_DICT)

pygame.init()
screen = pygame.display.set_mode((100, 100))  # black square
pygame.display.set_caption('Pygame Keyboard Test')
pygame.mouse.set_visible(0)

fileWaypoints = open(str(track_name)+".txt", "a")
x = 0
y = 0
z = 0
try:
    while True:
        keyState = pygame.key.get_pressed()
        steer_value = 1
        if(len(keyState) != 0):
            if keyState[pygame.K_UP]:
                throttle = 0.5
            elif keyState[pygame.K_DOWN]:
                throttle = -1.0
            else:
                throttle = 0.0
            if keyState[pygame.K_RIGHT]:
                steer = 1.0
            elif keyState[pygame.K_LEFT]:
                steer = -1.0
            else:
                steer = 0.0

            if keyState[pygame.K_x]:
                brake = 100.0
            else:
                brake = 0.0
            if keyState[pygame.K_c]:
                coordinates = str(x) + "," + str(y) + "," + str(z)
                fileWaypoints.write(coordinates)
                time.sleep(.3)
            
        observationReceived, tele, lidar, pos_x, pos_y, pos_z, speed, cte = gym_env.step(steer, throttle, brake, False)
        x = pos_x
        y = pos_y
        z = pos_z

except KeyboardInterrupt:
    #Resets steering and throttle to 0, applies breaks and resets for 1 second
    gym_env.step(0.0, 0.0, 100.0, False)
    time.sleep(1)
    print('\nCollection Stopped')
    gym_env.step(0.0, 0.0, 0.0, True)
