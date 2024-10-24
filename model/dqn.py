import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 定义Dueling DQN网络
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DuelingDQN, self).__init__()
        # 特征提取层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))

        # 计算卷积层后的输出大小
        self._create_hidden_layers(input_channels)

        # 价值流
        self.fc_value = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 优势流
        self.fc_advantage = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _create_hidden_layers(self, input_channels):
        # 使用虚拟输入计算卷积层后的输出大小
        dummy_input = torch.zeros(1, input_channels, 1, 21)  # 输入张量的形状
        x = self.relu(self.conv1(dummy_input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        self.conv_output_size = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        q_values = value + advantage - advantage.mean()
        return q_values

# 代理类
class Agent:
    def __init__(self, state_shape, num_actions):
        input_channels = state_shape[0]
        self.num_actions = num_actions
        self.policy_net = DuelingDQN(input_channels, num_actions)
        self.target_net = DuelingDQN(input_channels, num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(50000)  # 记忆库容量为50,000
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)  # 学习率为0.001
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000
        self.batch_size = 128  # 批量大小为128
        self.gamma = 0.95  # 折扣因子为0.95

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_next_state = torch.cat(batch_next_state)
        batch_done = torch.cat(batch_done)

        state_action_values = self.policy_net(batch_state).gather(1, batch_action)

        with torch.no_grad():
            next_state_actions = self.policy_net(batch_next_state).max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(batch_next_state).gather(1, next_state_actions).squeeze()

        expected_state_action_values = (next_state_values * self.gamma * (1 - batch_done)) + batch_reward

        loss = nn.MSELoss()(state_action_values.squeeze(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def soft_update(self, tau=0.01):
        # 软更新目标网络参数
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# 经验回放池
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
