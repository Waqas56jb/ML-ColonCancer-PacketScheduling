import numpy as np
import random
import gym
from gym import spaces
from collections import deque

# Constants for queues
VIDEO = 0
VOICE = 1
BEST_EFFORT = 2

# QoS delay requirements
DELAY_REQUIREMENTS = {
    VIDEO: 6,
    VOICE: 4,
    BEST_EFFORT: 9999  # Best-effort, no strict delay
}

ARRIVAL_RATES = {
    VIDEO: 0.3,
    VOICE: 0.25,
    BEST_EFFORT: 0.4
}

class RouterEnv(gym.Env):
    def __init__(self, switch_penalty=False):
        super(RouterEnv, self).__init__()

        self.switch_penalty = switch_penalty
        self.max_queue_length = 50

        # 3 queues
        self.queues = [deque() for _ in range(3)]

        # Time
        self.time = 0

        # Last queue served (for switch penalty)
        self.current_queue = -1

        # Observation: queue sizes and head packet delays
        self.observation_space = spaces.Box(low=0, high=self.max_queue_length,
                                            shape=(6,), dtype=np.int32)
        # Action: 0 - Video, 1 - Voice, 2 - Best-effort
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.time = 0
        self.queues = [deque() for _ in range(3)]
        self.current_queue = -1
        return self._get_state()

    def step(self, action):
        reward = 0
        done = False

        # Packet arrival process
        for i in range(3):
            if random.random() < ARRIVAL_RATES[i]:
                self.queues[i].append(self.time)

        # Handle switch penalty
        if self.switch_penalty and action != self.current_queue:
            self.current_queue = action
            self.time += 1
            return self._get_state(), -1, done, {}

        self.current_queue = action

        # Send packet from selected queue
        if self.queues[action]:
            arrival_time = self.queues[action].popleft()
            delay = self.time - arrival_time

            # QoS Reward
            if delay <= DELAY_REQUIREMENTS[action]:
                reward = 1
            else:
                reward = -1
        else:
            reward = -0.5  # Penalize for selecting empty queue

        self.time += 1
        return self._get_state(), reward, done, {}

    def _get_state(self):
        state = []
        for q in self.queues:
            state.append(len(q))
            if q:
                state.append(self.time - q[0])
            else:
                state.append(0)
        return np.array(state, dtype=np.int32)

# Simple DQN Agent
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state):
        self.memory.append(Transition(state, action, next_state, reward))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch))

        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)

        q_values = self.model(state_batch).gather(1, action_batch).squeeze()
        next_q_values = self.target_model(next_state_batch).max(1)[0].detach()
        target = reward_batch + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop
if __name__ == '__main__':
    env = RouterEnv(switch_penalty=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episodes = 500
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.replay(32)
        agent.update_target_model()
        print(f"Episode {e + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")