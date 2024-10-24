import random
from collections import deque

# 经验回放池
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # 使用deque实现的固定容量队列

    def push(self, *args):
        """将经验存储到回放池"""
        self.memory.append(tuple(args))

    def sample(self, batch_size):
        """随机采样一批经验"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """返回当前存储的经验数量"""
        return len(self.memory)
