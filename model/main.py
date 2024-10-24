from traffic_environment import TrafficEnvironment
from dueling_dqn import Agent
import torch
import matplotlib.pyplot as plt

def train_agent():
    num_episodes = 10  # 回合数
    env = TrafficEnvironment()
    agent = Agent(state_shape=(28, 1, 21), num_actions=8)

    # 记录每个回合的平均等待时间和总排队长度
    avg_waiting_times = []
    total_queue_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0  # 记录累计奖励
        total_waiting_time = 0  # 累计等待时间
        total_queue_length = 0  # 累计排队长度

        for t in range(4500):  # 每个回合的最大时间步数
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            reward = torch.tensor([reward], dtype=torch.float)
            done = torch.tensor([done], dtype=torch.float)

            # 存储过渡
            agent.memory.push(state, action, reward, next_state, done)

            # 转移到下一个状态
            state = next_state

            # 执行优化
            agent.optimize_model()

            # 软更新目标网络
            agent.soft_update(tau=0.01)

            total_reward += reward.item()

            # 记录等待时间和排队长度
            total_waiting_time += env.get_total_waiting_time()
            total_queue_length += env.get_total_queue_length()

            if done:
                break

        # 计算平均等待时间和总排队长度
        avg_waiting_time = total_waiting_time / (t + 1)
        avg_waiting_times.append(avg_waiting_time)
        total_queue_lengths.append(total_queue_length)

        print(f"Episode {episode} completed. Total reward: {total_reward}, Avg Waiting Time: {avg_waiting_time:.2f}, Total Queue Length: {total_queue_length}")

    # 绘制等待时间和排队长度的变化曲线
    plot_metrics(avg_waiting_times, total_queue_lengths)

def plot_metrics(avg_waiting_times, total_queue_lengths):
    episodes = range(len(avg_waiting_times))

    plt.figure(figsize=(12, 5))

    # 绘制平均等待时间
    plt.subplot(1, 2, 1)
    plt.plot(episodes, avg_waiting_times, label='Average Waiting Time')
    plt.xlabel('Episode')
    plt.ylabel('Average Waiting Time')
    plt.title('Average Waiting Time over Episodes')
    plt.legend()

    # 绘制总排队长度
    plt.subplot(1, 2, 2)
    plt.plot(episodes, total_queue_lengths, label='Total Queue Length', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Queue Length')
    plt.title('Total Queue Length over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_agent()
