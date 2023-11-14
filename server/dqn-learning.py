import copy
import numpy as np
import os
import torch
import collections
import websockets
import asyncio
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
from time import time
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import random

"""
TODOS: 
- Clean up code, make sure everything on right device, make sure preprocessing is correct and happens in model not environment. 
- Send actions as actual integers, not strings.
- Add no grad where necessary
- Add a way to stop when copying over the model.
"""

action_map = {0: "left", 1: "right", 2: "forward"}

# CONSTANTS
IP = "172.26.177.26"
# PORT = 48622
PORT = 8765
THRESHOLD = 50
# HYPERPARAMETERS
ACTION_SPACE = len(action_map)
UPDATE_FREQ = 50
E = 300
EPSILON = 0.1
LEARNING_RATE = 0.001
GAMMA = 0.9
BATCH_SIZE = 32
MAX_EPISODE_LEN = 300
STATE_BYTES = 224 * 224 * 3


# Replay Memory Class
class Replay_Memory:
    def __init__(self, memory_size=50000, burn_in=10):
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
        super(CNN, self).__init__()
        # Load the pretrained MobileNetV2 model
        self.alexNetv2 = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "alexnet",
            pretrained=True,
        )

        # Freeze all layers
        # for param in self.alexNetv2.parameters():
        #     param.requires_grad = False

        # Modify the classifier (fully connected) layer to match the desired output size
        num_features = self.alexNetv2.classifier[1].in_features
        self.alexNetv2.classifier = nn.Sequential(
            nn.Linear(num_features, 128),  # You can adjust the size of the hidden layer
            nn.ReLU(),
            nn.Linear(128, action_size),  # 5 output units for 5 actions
        )

        # self.preprocess = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )

    def forward(self, state):
        if self.device is not None:
            state = state.to(self.device)

        return self.alexNetv2(state)


class QNetwork:
    def __init__(self, lr=0.001, load_model=False, model_path=None):
        if not load_model:
            self.model = CNN(ACTION_SPACE)
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
            else:
                mps_device = torch.device("mps")
                self.model.to(mps_device)
                self.model.device = mps_device
            self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        else:
            self.load_model(model_path)

    def save_model(self, path="model.pt"):
        torch.save(self.model, path)

    def load_model(self, path):
        raise NotImplementedError


class DQN_Agent:
    def __init__(
        self, E, episolon, lr, gamma, batch_size, env, train=True, model_path=None
    ):
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
        self.loss_fn = nn.MSELoss()

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.random() < self.episolon:
            return np.random.randint(0, ACTION_SPACE)
        else:
            return torch.argmax(q_values).item()

    def greedy_policy(self, q_values):
        return torch.argmax(q_values).item()

    def optimize_model(self):
        # Need to fix this!
        batch = self.replay_memory.sample_batch(
            self.batch_size
        )  # returns a list of tuples of length batch_size
        y_i = np.zeros((self.batch_size, ACTION_SPACE))
        y_f = np.zeros((self.batch_size, ACTION_SPACE))
        # TODO: Could Parallelize this, not sure if doing this correctly
        for i, (state, action, reward, next_state, end) in enumerate(batch):
            state = state.requires_grad_(True)
            q_values = self.q_w.model(state)[0]

            q_values_next = self.q_target.model(next_state)[0]

            if end:
                y_i[i][action] = reward
            else:
                y_i[i][action] = reward + self.gamma * torch.max(q_values_next)

            y_f[i][action] = q_values[action].item()

        y_i = torch.tensor(y_i)
        y_f = torch.tensor(y_f).requires_grad_(True)

        self.q_w.model.optimizer.zero_grad()
        self.loss_fn(y_f, y_i).backward()
        self.q_w.model.optimizer.step()

        print("optimized")

    async def train(self):
        # Training the agent
        starting_state = await self.env.start()
        await self.burn_in_memory(starting_state)

        c = 0
        episode_len = 0
        for i in range(self.E):
            print("episode", i)
            state = await self.env.reset()
            while True:
                q_values = self.q_w.model(
                    state
                )  # returns a tensor of length ACTION_SPACE
                action = self.epsilon_greedy_policy(
                    q_values
                )  # returns an int corresponding to action to take
                next_state, reward, terminal_state = await self.env.step(
                    action
                )  # returns a tuple of (state, reward, end)
                self.replay_memory.append(
                    (state, action, reward, next_state, terminal_state)
                )  # appends tuple to replay memory

                state = next_state

                self.optimize_model()

                c += 1
                episode_len += 1

                if c % UPDATE_FREQ == 0:
                    self.q_target = copy.deepcopy(self.q_w)

                if episode_len >= MAX_EPISODE_LEN or terminal_state:
                    break

    async def burn_in_memory(self, starting_state):
        state = starting_state
        print("starting burn in")
        for _ in range(self.replay_memory.burn_in):
            action = np.random.randint(0, ACTION_SPACE)
            next_state, reward, done = await self.env.step(action)
            self.replay_memory.append((state, action, reward, next_state, done))
            if done:
                state = await self.env.reset()
            else:
                state = next_state

        print("end burn in")

    def testNetwork(self):
        for _ in tqdm(range(1000)):
            state = torch.randn(1, 3, 224, 224)
            q_values = self.q_w.model(state)
            _ = self.greedy_policy(q_values)

    async def test_connection(self):
        initial_state = await self.env.start()
        q_values = self.q_w.model(initial_state)
        print(time(), q_values)
        end = False
        for i in range(100):
            print("episode", i)
            while not end:
                # Choose random int from 1 to 3 inclusive
                action = random.choice([0, 1, 2])
                next_state, reward, term = await self.env.step(action)
                q_values = self.q_w.model(next_state)
                print(time(), q_values)
                end = term

            await self.env.reset()
            end = False


class EnvironmentCommunicator:
    def __init__(self, IP, PORT):
        # Initialize websocket that communicates with car.
        # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket.connect((IP, PORT))
        self.IP = IP
        self.PORT = PORT
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(f"ws://{self.IP}:{self.PORT}")

    async def step(self, action):
        """
        returns (state, reward, done)
        """
        # Send the action to the car
        # self.socket.sendall(action_map[action].encode())
        await self.websocket.send(str(action).encode())

        # The car should take the action, and then send back the new state and greyscale readings.
        msg = await self.websocket.recv()

        # Do some modification to get state into torch tensor, and greyscale to reward.
        state = msg[:STATE_BYTES]
        state = self.bytes_to_tensor(state)

        grey_scale_and_terminal = msg[STATE_BYTES:]

        # Turn grey_scale_and_terminal into list of ints
        grey_scale = list(grey_scale_and_terminal[:3])
        terminal = bool(grey_scale_and_terminal[3])
        # print(grey_scale, terminal)
        # each grayscale reading is 1 byte, there are 3 readings, turn to list
        if terminal:
            reward = 0
        else:
            if any([r < THRESHOLD for r in grey_scale]):
                reward = 0
            else:
                reward = 1
        return state, reward, terminal

    async def reset(self):
        # Tell the car to reset.
        await self.websocket.send("reset".encode())
        # Wait for the car to reset, and send back starting state.
        state = await self.websocket.recv()
        state = self.bytes_to_tensor(state)

        return state

    def bytes_to_tensor(self, bytes):
        state_np = np.frombuffer(bytes, dtype=np.uint8).reshape(224, 224, 3)
        state_tensor = self.normalize(
            torch.tensor(state_np).permute(2, 0, 1).float() / 255.0
        ).unsqueeze(0)
        return state_tensor

    async def start(self):
        # Tell the car to get ready to start receiving communications.
        await self.websocket.send("start".encode())
        state = await self.websocket.recv()
        state = self.bytes_to_tensor(state)
        return state

    def close_connection(self):
        self.websocket.close()


async def main():
    # Initialize environment
    env = EnvironmentCommunicator(IP, PORT)
    await env.connect()
    agent = DQN_Agent(E, EPSILON, LEARNING_RATE, GAMMA, BATCH_SIZE, env)
    # await agent.test_connection()
    await agent.train()
    agent.q_w.save_model("model.pt")
    # agent.testNetwork()

    # agent.train()


if __name__ == "__main__":
    asyncio.run(main())
    # main()
