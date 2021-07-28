import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

###################################################################################
# @class: Model_Net
# This is the model that will be used for training
###################################################################################
class Model_Net(nn.Module):
    # Initialize the Model
    def __init__(self):
        super(Model_Net, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 3)

    # Forward Pass the image
    def forward(self, x):
      x = self.model(x[None,...])
      return x

###################################################################################
# @class: Model_Trainer
# Function that trains the models
###################################################################################
class Model_Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # Learning Rate
        self.gamma = gamma # Gamme value
        self.model = model # Model passed in for training
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Optimizer
        self.criterion = nn.MSELoss() # Type of Loss Function


    #############################################################################
    # Train the model
    # @param: state - Type: Image - State (Images)
    # @param: action - Type: Image - Action taken at the State
    # @param: Reward - Type: Float - Reward based on the state + action
    # @param: next_state - Type: Image - New Image (Car location) based on State + Action
    # @param: done - Type: Boolean - If state + action caused the car to go into a game over state or not
    #############################################################################
    def train_step(self, state, action, reward, next_state, done):
        # Convert into Torch Tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # If necessary, convert into a batch so model can predict
        # if len(state.shape) == 3 and len(state[0].shape) == 2:
        state_unsqueeze = torch.unsqueeze(state, 0)
        next_state_unsqueeze = torch.unsqueeze(next_state, 0)
        action_unsqueeze = torch.unsqueeze(action, 0)
        reward_unsqueeze = torch.unsqueeze(reward, 0)
        done_unsqueeze = (done, )
        
        pred = self.model(state_unsqueeze) # Models Prediction on State
        target = pred.clone() # CLone the Prediction
        for idx in range(len(done_unsqueeze)):
            Q_new = reward_unsqueeze[idx]
            if not done_unsqueeze[idx]: # Use all the non-game over moves for training
                # Take reward given by state + action and add it to the model's prediction
                Q_new = reward_unsqueeze[idx] + self.gamma * torch.max(self.model(next_state_unsqueeze[idx][None, ...]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new # Set the Target prediction to the new Q_value

        self.optimizer.zero_grad() # Zero out Gradients
        loss = self.criterion(target, pred) # Determine the Loss
        loss.backward() # Back-propagate
        self.optimizer.step() # Perform Gradient Updates
