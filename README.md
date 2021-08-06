# Reinforcement-Learning-Autonomous-Vehicle
Reinforcement Learning To Create An Autonomous Vehicle On The Donkey-Gym Environment.

How to use this RL Car.
1. Set the vehicle settings accordingly in the gym_wrapper.py.
2. Change the location of the waypoint .txt file in gym_wrapper.py
3. Create your own reward function in Reward.py
4. Create the model architecture to be used in model.py
5. Modify the number of games/Learning Rate/Batch size in agent.py
6. Choose the amount of games you will train for before evaluating the model by modifying the train_ratio value in main.py

## List of all Files
- Agent.py
  - The Agent is considered your car. This file will initialize the model, the memory for training, as well select actions for your car to take.   
- gym_wrapper.py
  - This file contains the wrapper for the gyminterface.py file. This allows for communication between the simulator and your agent. In the __init__() function, this is where you can customize your vehicle settings as well as server connections.
- gyminterface.py
  - This file communicates with the simulator and sends/receives raw data to be utilized and processed by the gym_wrapper.py class.
- main.py
  - This is where the training loop is located and what initializes everything. 
- model.py
  - This is where users can input their own PyTorch-based model that will be utilized in the RL process. 
- Reward.py
  - This file allows you to customize the reward function. Higher reward = good behavior and lower reward = bad behavior. 
- TelemetryPack.py
  - This file is used in coordination with gyminterface for raw data communication. 
- way_points.py
  This file loads in the selected waypoint coordinates from a .txt file and can be used in teleportation or reward functions.
  
 
  


Credit:

(Triton-Racer-Sim)[https://github.com/Triton-AI/Triton-Racer-Sim] for the base gyminterface.py and telemetryPack.py files.

(Python-Engineer)[https://github.com/python-engineer] - Code based off his implementation of Snake AI
