import torch
import random
import torchvision
import numpy as np
from collections import deque
from gym_wrapper import DonkeySimEnv
from model import Model_Net, Model_Trainer

MAX_MEMORY = 100000
BATCH_SIZE = 64
LR = 0.1


class Agent:

    def __init__(self):
        self.n_games = 0
        self.games = 100
        self.epslion = 0  # Random
        self.gamma = 0.9  # Discount Rate / Smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY) # Will pop data if length because greater than max length
        self.model_train = torchvision.models.resnet18(pretrained=True)
        self.model_train.fc = torch.nn.Linear(512, 3)
        self.trainer = Model_Trainer(model=self.model_train, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return np.moveaxis(game.get_state(), -1, 0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        print("Training...")
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        for states, actions, rewards, next_states, dones in mini_sample:
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, evaluation=False):
        # Random move = Exploration = Compare to epilson and random move
        # Model move = Exploitation = Prediction
        self.epslion = self.games - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 10) < 4:
            steering = (random.randint(0, 2))
            final_move[steering] = 1
        else:
            state_current = torch.tensor(state, dtype=torch.float)
            state_current = torch.unsqueeze(state_current, 0)
            prediction = self.model_train(state_current)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def model_save(self):
        torch.save(self.model_train.state_dict(),"Model_Trained.pth")


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = DonkeySimEnv()
    counter = 1
    while agent.n_games < 200:
        state_old = agent.get_state(game) # Get Old State
        final_move = agent.get_action(state_old) # Get Move
        obs, reward, done, info, score = game.step(final_move) # Perform action
        if score > record:
            agent.epslion = 200
        state_new = agent.get_state(game) # Get new state
        agent.train_short_memory(state_old, final_move, reward, state_new, done) # Train short memory
        agent.remember(state_old, final_move, reward, state_new, done) # Remember 
        if done: # Train long term memory
            counter += 1
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model_save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
