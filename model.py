import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Model_Net(nn.Module):
    def __init__(self):
        super(Model_Net, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

    def forward(self, x):
      x = self.model(x[None,...])
      return x

    def save(self, file_name='model.pth'):
        torch.save(self.model.state_dict(), file_name)


class Model_Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 3 and len(state[0].shape) == 2:
            # (1, x)
            # print("YES")
            state_unsqueeze = torch.unsqueeze(state, 0)
            next_state_unsqueeze = torch.unsqueeze(next_state, 0)
            action_unsqueeze = torch.unsqueeze(action, 0)
            reward_unsqueeze = torch.unsqueeze(reward, 0)
            done_unsqueeze = (done, )
            
        pred = self.model(state_unsqueeze)

        target = pred.clone()
        for idx in range(len(done_unsqueeze)):
            Q_new = reward_unsqueeze[idx]
            if not done_unsqueeze[idx]:
                Q_new = reward_unsqueeze[idx] + self.gamma * \
                    torch.max(self.model(next_state_unsqueeze[idx][None, ...]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
