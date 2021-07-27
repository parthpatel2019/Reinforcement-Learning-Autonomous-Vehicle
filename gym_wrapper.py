import numpy as np
import time
import gym_donkeycar
import gym


class DonkeySimEnv:

    def __init__(self):
        exe_path = f"./donkey_sim.exe"
        port = 9091
        conf = {
            "exe_path": exe_path,
            "host": "127.0.0.1",
            "port": port,
            "body_style": "f1",
            "body_rgb": (0, 0, 0),
            "car_name": "Parth",
            "font_size": 100,
            "racer_name": "MyRLModel",
            "country": "USA",
            "bio": "RL Model",
            "max_cte": 5,
            "fov": 80,
            "fish_eye_x": 0.0,
            "fish_eye_y": 0.0,
            "img_w": 224,
            "img_h": 224,
            "img_d": 3,
            "img_enc": 'JPG',
            "offset_x": 0.0,  # sides
            "offset_y": 2.0,  # height #Jetsim 2.0
            "offset_z": 0.0,  # forward
            "rot_x": 20.0,  # tilt #Jetsim 0.0
            #"rot_y": 180,    #rotate
        }
        self.env = gym.make('donkey-circuit-launch-track-v0', conf=conf)
        self.current_time = time.time()
        obs, reward, done, info = self.env.step([0, 0])
        self.obs = obs
        self.restart = 1
        WayPointsClass = WayPoints(textFile='circuit_points.txt')
        self.waypoints = WayPointsClass.returnWayPoints()

    def distCalc(self, point, coord_x, coord_z):
        dist_z = np.power((coord_z - point.z), 2)
        dist_x = np.power((coord_x - point.x), 2)
        return np.sqrt(dist_z + dist_x)

    def step(self, action):
        if self.restart == 1:
            self.current_time = time.time()
            self.restart = 0
        # steering_dictionary = {
        #    0: -0.5,
        #    1: -0.4,
        #    2: -0.3,
        #    3: -0.2,
        #    4: -0.1,
        #    5: 0.0,
        #    6: 0.1,
        #    7: 0.2,
        #    8: 0.3,
        #    9: 0.4,
        #    10: 0.5
        # }
        steering_dictionary = {
            0: -0.4,
            1: 0,
            2: 0.4
        }
        total = [steering_dictionary[np.argmax(action)], 0.5]
        # print("Arguments:", total)
        obs, reward, done, info = self.env.step(total)
        self.obs = obs
        # new_reward = -1
        # if(info['hit'] == 'none'):
        #     new_reward += 0.75
        #     # print("Did not hit!")
        #     if(info['cte'] < 4 and info['cte'] > -4):
        #         new_reward += 0.75
        #         # print("In center")
        #         new_reward += ((time.time() - self.current_time))
        # # print("Reward: ", new_reward)
        new_reward = -2
        if(done == False):
            new_reward += 1
        for point in self.waypoints:
          if (point.hit == False):
            if(self.distCalc(point, float(info['pos'][0]), float(info['pos'][2])) < 1):
                new_reward += 1
                point.hit = True
                print("Hit Waypoint: ", point.x, " ", point.z)
        return obs, new_reward, done, info, (time.time() - self.current_time)

    def get_state(self):
        return self.obs

    def reset(self):
        obs = self.env.reset()
        for point in self.waypoints:
          point.hit = False
        self.restart = 1

    def close(self):
        self.current_time = None
        self.env.close()
