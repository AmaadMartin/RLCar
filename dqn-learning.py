import copy
import numpy as np
import os
import torch
import collections
import torch
import torchvision.models as models
import torch.nn.functional as F

action_map = {
    0: 'left',
    1: 'right',
    2: 'forward',
    3: 'backward',
    4: 'stop'
}

num_actions = len(action_map)
UPDATE_FREQ = 50

#Replay Memory Class
class Replay_Memory:
    def __init__(self, memory_size=50000, burn_in=1000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # Hint: you might find this useful:
        # 		collections.deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = collections.deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        batch = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            batch.append(self.buffer[idx])

        return batch

    def append(self, transition):
        # Appends transition to the memory.
        self.buffer.append(transition)


class CNN(torch.nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.feature_extractor = models.mobilenet_v2(pretrained=True)

        # Modify the feature extractor to accept 10x10 images
        self.feature_extractor.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.fc1 = torch.nn.Linear(6272, 128)
        self.fc2 = torch.nn.Linear(128, action_size)
    
    def forward(self, state):
        x = self.feature_extractor(state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QNetwork:
    def __init__(self, lr=0.001, load_model=False, model_path=None):
        if not load_model:
            global num_actions
            self.model = CNN(10, num_actions)
            self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError
    

class DQN_Agent:
    def __init__(self, E, episolon, lr, gamma, batch_size, env, train=True, model_path=None):
        # Initialize your agent's parameters
        self.q_w = QNetwork()
        self.q_target = copy.deepcopy(self.q_w)
        self.replay_memory = Replay_Memory()
        self.E = E
        self.episolon = episolon
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
    
    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.random() < self.episolon:
            return np.random.randint(0, num_actions)
        else:
            return np.argmax(q_values)
    
    def greedy_policy(self, q_values):
        return np.argmax(q_values)

    def train(self):
        # Training the agent
        loss_fn = torch.nn.MSELoss()
        self.burn_in_memory()
        c = 0
        for _ in range(self.E):
            state = self.env.reset()
            end = False
            while not end:
                q_values = self.q_w(state)
                action = self.epsilon_greedy_policy(q_values)
                next_state, reward, end = self.env.step(action)
                self.replay_memory.append((state, action, reward, next_state, end))

                batch = self.replay_memory.sample_batch(self.batch_size)
                y_i = np.zeros(self.batch_size)
                y_f = np.zeros(self.batch_size)

                #TODO: Could Parallelize this
                for i, (state, action, reward, next_state, end) in enumerate(batch):
                    q_values = self.q_w(state)
                    q_values_next = self.q_target(next_state)
                    # q_values_next = q_values_next.detach()
                    if end:
                        y_i[i] = reward
                    else:
                        y_i[i] = reward + self.gamma * torch.max(q_values_next)

                    y_f[i] = q_values[action]
                
                self.q_w.model.optimizer.zero_grad()
                loss_fn(y_f, y_i).backward()
                self.q_w.model.optimizer.step()
                c+=1

                if c % UPDATE_FREQ == 0:
                    self.q_target = copy.deepcopy(self.q_w)

    def burn_in_memory(self):
        state = self.env.reset()
        for _ in range(self.replay_memory.burn_in):
            action = np.random.randint(0, num_actions)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_memory.append((state, action, reward, next_state, done))
            if done:
                state = self.env.reset()
            else:
                state = next_state

        