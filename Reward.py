import numpy as np

MAX_DISTANCE_FROM_WAYPOINT = 1 # Radius that the car needs to be within of the way point to get the reward for it


###################################################################################
# Calculate the Euclidean Distance between a waypoint and the current coordinate
# @param: point - Type: Point - One of the way points passed in
# @param: x_coordinate - Type: Float - X-coordinate of the car
# @param: z_coordinate - Type: Float - Z-coordinate of the car
# @return: The euclidean distance between the car and the waypoint
###################################################################################

def distCalc(point, x_coordinate, z_coordinate):
    dist_z = np.power((z_coordinate - point.z), 2)
    dist_x = np.power((x_coordinate - point.x), 2)
    return np.sqrt(dist_z + dist_x)


###################################################################################
# The Reward Function. 
# @param: observationReceived - Type: Image - Image seen from the simulator
# @param: done - Type: Boolean - Default done given from simulator. If car has hit an object or current cte > max cte, done = True
# @param: info - Type: Json - Contains: cte, (x, y, z) coordinates, speed, and hit 
# @return: The Reward
###################################################################################
def getReward(observationReceived, tele, done, waypoints):
    new_reward = -100
    if(done == False):
        new_reward = 0
        for point in waypoints:  # Cycle through way points and see if car is within radius of it
            if (point.hit == False):  # If the waypoint has not been hit on the current iteration
                if(distCalc(point, float(tele.pos_x), float(tele.pos_z)) < MAX_DISTANCE_FROM_WAYPOINT):
                    new_reward += 60
                    point.hit = True
                    print("Hit Waypoint: ", point.x, " ", point.z)
        if (tele.cte > -0.9 and tele.cte < 0.4):
            new_reward += 100
    return new_reward
