import numpy as np
import random
import torch

# 简化的仿真环境
class TrafficEnvironment:
    def __init__(self):
        self.num_directions = 4  # 北、南、东、西
        self.num_lanes_per_direction = 3
        self.num_grids = 21
        self.max_speed = 13.9  # 最大速度（m/s）
        self.signal_phases = 4  # 信号相位数量
        self.current_phase = 0  # 当前信号相位索引
        self.phase_duration = 0  # 当前相位持续时间
        self.phase_times = [5, 10, 15, 20, 30, 35, 40, 45]  # 可选的绿灯持续时间（秒）
        self.waiting_times = np.zeros((self.num_directions * self.num_lanes_per_direction, self.num_grids))
        self.reset()

    def reset(self):
        # 初始化环境状态
        self.position_matrices = np.zeros((self.num_directions * self.num_lanes_per_direction, self.num_grids))
        self.velocity_matrices = np.zeros((self.num_directions * self.num_lanes_per_direction, self.num_grids))
        self.waiting_times = np.zeros_like(self.position_matrices)  # 重置等待时间矩阵
        self.current_phase = 0
        self.phase_duration = 0
        self.generate_random_traffic()
        return self.get_state()

    def generate_random_traffic(self):
        # 随机生成车辆位置和速度
        for lane_index in range(self.position_matrices.shape[0]):
            num_vehicles = random.randint(0, self.num_grids)
            vehicle_positions = random.sample(range(self.num_grids), num_vehicles)
            for pos in vehicle_positions:
                self.position_matrices[lane_index, pos] = 1
                self.velocity_matrices[lane_index, pos] = random.uniform(0, self.max_speed) / self.max_speed  # 归一化

    def step(self, action):
        # 执行动作，更新环境状态
        green_duration = self.phase_times[action.item()]
        self.phase_duration = green_duration

        # 更新车辆状态（绿灯期间）
        self.update_traffic(is_yellow=False)

        # 添加3秒黄灯期间的车辆更新
        self.phase_duration = 3  # 黄灯持续3秒
        self.update_traffic(is_yellow=True)

        # 切换到下一个相位
        self.current_phase = (self.current_phase + 1) % self.signal_phases

        # 获取奖励
        reward = self.calculate_reward()
        # 获取下一个状态
        next_state = self.get_state()
        # 判断是否结束
        done = False  # 在此示例中，我们不设置终止条件
        return next_state, reward, done

    def update_traffic(self, is_yellow=False):
        move_prob_green = 0.7
        move_prob_red = 0.1
        move_prob_yellow = 0.4  # 黄灯期间的车辆移动概率

        for _ in range(int(self.phase_duration)):
            new_position_matrices = np.zeros_like(self.position_matrices)
            new_velocity_matrices = np.zeros_like(self.velocity_matrices)
            new_waiting_times = np.zeros_like(self.waiting_times)
            for lane_index in range(self.position_matrices.shape[0]):
                for grid_index in range(self.num_grids):
                    if self.position_matrices[lane_index, grid_index] == 1:
                        # 判断是否可以前进
                        if grid_index > 0 and self.position_matrices[lane_index, grid_index - 1] == 0:
                            if is_yellow:
                                move_prob = move_prob_yellow
                            else:
                                move_prob = move_prob_green if self.is_green_light(lane_index) else move_prob_red
                            if random.random() < move_prob:
                                # 前进一格
                                new_position_matrices[lane_index, grid_index - 1] = 1
                                new_velocity_matrices[lane_index, grid_index - 1] = self.velocity_matrices[lane_index, grid_index]
                                new_waiting_times[lane_index, grid_index - 1] = self.waiting_times[lane_index, grid_index]
                            else:
                                # 保持不动，等待时间增加
                                new_position_matrices[lane_index, grid_index] = 1
                                new_velocity_matrices[lane_index, grid_index] = 0
                                new_waiting_times[lane_index, grid_index] = self.waiting_times[lane_index, grid_index] + 1
                        else:
                            # 前方有车或到达路口，无法前进
                            new_position_matrices[lane_index, grid_index] = 1
                            new_velocity_matrices[lane_index, grid_index] = 0
                            new_waiting_times[lane_index, grid_index] = self.waiting_times[lane_index, grid_index] + 1
            self.position_matrices = new_position_matrices
            self.velocity_matrices = new_velocity_matrices
            self.waiting_times = new_waiting_times

    def is_green_light(self, lane_index):
        # 判断当前车道是否为绿灯
        direction = lane_index // self.num_lanes_per_direction
        return direction == self.current_phase

    def calculate_reward(self):
        # 使用排队长度的负值作为奖励，鼓励减少排队长度
        total_queue_length = np.sum(self.position_matrices)
        reward = -total_queue_length
        return reward

    def get_state(self):
        # 保留每条车道的信息，不对车道进行求和
        position_matrices = self.position_matrices.reshape(self.num_directions * self.num_lanes_per_direction, 1, self.num_grids)
        velocity_matrices = self.velocity_matrices.reshape(self.num_directions * self.num_lanes_per_direction, 1, self.num_grids)

        # 信号相位向量，形状为 (4, 1, 1)
        signal_phase_vector = np.zeros((4, 1, 1))
        signal_phase_vector[self.current_phase, 0, 0] = 1

        # 在网格维度上重复信号相位向量，得到形状为 (4, 1, 21)
        signal_phase_matrix = np.repeat(signal_phase_vector, self.num_grids, axis=2)

        # 在通道维度上堆叠，得到形状为 (12 + 12 + 4, 1, 21) = (28, 1, 21)
        state = np.concatenate((position_matrices, velocity_matrices, signal_phase_matrix), axis=0)
        # 转换为 PyTorch 张量，形状为 (
