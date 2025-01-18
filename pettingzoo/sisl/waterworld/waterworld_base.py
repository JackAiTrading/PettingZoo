import math

import gymnasium
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.spatial import distance as ssd

from pettingzoo.sisl.waterworld.waterworld_models import (
    Evaders,
    Obstacle,
    Poisons,
    Pursuers,
)

FPS = 15


class WaterworldBase:
    """水世界环境类。

    该环境模拟了一个水世界场景，其中有追捕者、逃避者、毒物和障碍物。

    Attributes:
        pixel_scale (int): 像素尺寸
        clock (pygame.time.Clock): pygame时钟
        FPS (int): 帧率
        handlers (list): 碰撞处理器列表
        n_coop (int): 捕获食物粒子所需的代理数量
        n_evaders (int): 食物粒子数量
        n_obstacles (int): 障碍物数量
        n_poisons (int): 毒物数量
        n_pursuers (int): 代理数量
        n_sensors (int): 每个代理的传感器数量
        base_radius (float): 代理半径
        obstacle_radius (float): 障碍物半径
        sensor_range (float): 传感器范围
        pursuer_speed (float): 代理最大速度
        evader_speed (float): 食物粒子最大速度
        poison_speed (float): 毒物最大速度
        speed_features (bool): 是否在状态空间中包含实体速度
        pursuer_max_accel (float): 代理最大加速度
        encounter_reward (float): 遇到食物粒子的奖励
        food_reward (float): 获取食物粒子的奖励
        local_ratio (float): 本地奖励与全局奖励的比例
        poison_reward (float): 获取毒物的奖励（或惩罚）
        thrust_penalty (float): 大动作的负面奖励的缩放因子
        max_cycles (int): 最大循环次数
        control_rewards (list): 控制奖励列表
        behavior_rewards (list): 行为奖励列表
        last_dones (list): 上一步结束标志列表
        last_obs (list): 上一步观察列表
        last_rewards (list): 上一步奖励列表
        initial_obstacle_coord (list): 初始障碍物坐标
        render_mode (str): 渲染模式
        screen (pygame.Surface): pygame窗口
        frames (int): 帧数
        num_agents (int): 代理数量
        observation_space (list): 观察空间列表
        action_space (list): 动作空间列表
    """

    def __init__(
        self,
        n_pursuers=2,
        n_evaders=5,
        n_poisons=10,
        n_obstacles=1,
        n_coop=1,
        n_sensors=30,
        sensor_range=0.2,
        radius=0.015,
        obstacle_radius=0.1,
        obstacle_coord=[(0.5, 0.5)],
        pursuer_max_accel=0.5,
        pursuer_speed=0.2,
        evader_speed=0.1,
        poison_speed=0.1,
        poison_reward=-1.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.5,
        local_ratio=1.0,
        speed_features=True,
        max_cycles=500,
        render_mode=None,
        FPS=FPS,
    ):
        """输入关键字参数。

        n_pursuers: 代理数量
        n_evaders: 食物粒子数量
        n_poisons: 毒物数量
        n_obstacles: 障碍物数量
        n_coop: 捕获食物粒子所需的代理数量
        n_sensors: 每个代理的传感器数量
        sensor_range: 传感器范围
        radius: 代理半径
        obstacle_radius: 障碍物半径
        obstacle_coord: 障碍物坐标，形如[n_obstacles, 2]的数组，值域为(0, 1)
        pursuer_max_accel: 代理最大加速度
        pursuer_speed: 代理最大速度
        evader_speed: 食物粒子最大速度
        poison_speed: 毒物最大速度
        poison_reward: 获取毒物的奖励（或惩罚）
        food_reward: 获取食物粒子的奖励
        encounter_reward: 遇到食物粒子的奖励
        thrust_penalty: 大动作的负面奖励的缩放因子
        local_ratio: 本地奖励与全局奖励的比例
        speed_features: 是否在状态空间中包含实体速度
        """
        self.pixel_scale = 30 * 25
        self.clock = pygame.time.Clock()
        self.FPS = FPS  # Frames Per Second

        self.handlers = []

        self.n_coop = n_coop
        self.n_evaders = n_evaders
        self.n_obstacles = n_obstacles
        self.n_poisons = n_poisons
        self.n_pursuers = n_pursuers
        self.n_sensors = n_sensors

        self.base_radius = radius
        self.obstacle_radius = obstacle_radius
        self.sensor_range = sensor_range

        self.pursuer_speed = pursuer_speed * self.pixel_scale
        self.evader_speed = evader_speed * self.pixel_scale
        self.poison_speed = poison_speed * self.pixel_scale
        self.speed_features = speed_features

        self.pursuer_max_accel = pursuer_max_accel

        self.encounter_reward = encounter_reward
        self.food_reward = food_reward
        self.local_ratio = local_ratio
        self.poison_reward = poison_reward
        self.thrust_penalty = thrust_penalty

        self.max_cycles = max_cycles

        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.behavior_rewards = [0 for _ in range(self.n_pursuers)]

        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = [None for _ in range(self.n_pursuers)]
        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]

        if obstacle_coord is not None and len(obstacle_coord) != self.n_obstacles:
            raise ValueError("obstacle_coord的长度与n_obstacles不匹配")
        else:
            self.initial_obstacle_coord = obstacle_coord

        self.render_mode = render_mode
        self.screen = None
        self.frames = 0
        self.num_agents = self.n_pursuers
        self.get_spaces()
        self._seed()

    def get_spaces(self):
        """定义所有代理的动作和观察空间。

        动作空间：2维连续空间，范围[-1,1]
        观察空间：(8*n_sensors+2)维连续空间，包含：
        - 传感器读数（位置和速度）
        - 是否与食物碰撞
        - 是否与毒物碰撞
        """
        if self.speed_features:
            obs_dim = 8 * self.n_sensors + 2
        else:
            obs_dim = 5 * self.n_sensors + 2

        obs_space = spaces.Box(
            low=np.float32(-np.sqrt(2)),
            high=np.float32(np.sqrt(2)),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        act_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = [obs_space for i in range(self.n_pursuers)]
        self.action_space = [act_space for i in range(self.n_pursuers)]

    def _seed(self, seed=None):
        """设置随机种子。

        参数：
            seed: 随机种子值

        返回：
            [seed]: 使用的随机种子列表
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obj(self):
        """创建所有移动对象实例。

        创建：
        - 追捕者（智能体）
        - 逃避者（食物）
        - 毒物
        - 障碍物
        """
        self.pursuers = []
        self.evaders = []
        self.poisons = []
        self.obstacles = []

        for i in range(self.n_pursuers):
            x, y = self._generate_coord(self.base_radius)
            self.pursuers.append(
                Pursuers(
                    x,
                    y,
                    self.pursuer_max_accel,
                    self.pursuer_speed,
                    radius=self.base_radius,
                    collision_type=i + 1,
                    n_sensors=self.n_sensors,
                    sensor_range=self.sensor_range,
                    speed_features=self.speed_features,
                )
            )

        for i in range(self.n_evaders):
            x, y = self._generate_coord(2 * self.base_radius)
            vx, vy = (
                (2 * self.np_random.random(1) - 1) * self.evader_speed,
                (2 * self.np_random.random(1) - 1) * self.evader_speed,
            )
            self.evaders.append(
                Evaders(
                    x,
                    y,
                    vx[0],
                    vy[0],
                    radius=2 * self.base_radius,
                    collision_type=i + 1000,
                    max_speed=self.evader_speed,
                )
            )

        for i in range(self.n_poisons):
            x, y = self._generate_coord(0.75 * self.base_radius)
            vx, vy = (
                (2 * self.np_random.random(1) - 1) * self.poison_speed,
                (2 * self.np_random.random(1) - 1) * self.poison_speed,
            )
            self.poisons.append(
                Poisons(
                    x,
                    y,
                    vx[0],
                    vy[0],
                    radius=0.75 * self.base_radius,
                    collision_type=i + 2000,
                    max_speed=self.poison_speed,
                )
            )

        for _ in range(self.n_obstacles):
            self.obstacles.append(
                Obstacle(
                    self.pixel_scale / 2,
                    self.pixel_scale / 2,
                    radius=self.obstacle_radius,
                )
            )

    def close(self):
        """关闭环境，清理资源。

        关闭pygame窗口并释放相关资源。
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def convert_coordinates(self, value, option="position"):
        """将pymunk坐标转换为pygame坐标。

        pygame坐标系：
                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
                        |       |                           │
                        |       |                           │
        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↓ y
        pymunk坐标系：
        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↑ y
                        |       |                           │
                        |       |                           │
                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x

        参数：
            value: 要转换的坐标值
            option: 转换类型，可以是"position"或"velocity"
        """
        if option == "position":
            return int(value[0]), self.pixel_scale - int(value[1])

        if option == "velocity":
            return value[0], -value[1]

    def _generate_coord(self, radius):
        """生成一个随机坐标，使得对象不与障碍物碰撞。

        参数：
            radius: 对象半径

        返回：
            coord: 生成的随机坐标
        """
        # 在[0, pixel_scale]范围内随机生成坐标(x, y)
        coord = self.np_random.random(2) * self.pixel_scale

        # 如果太接近障碍物，则重新生成
        for obstacle in self.obstacles:
            x, y = obstacle.body.position
            while (
                ssd.cdist(coord[None, :], np.array([[x, y]]))
                <= radius * 2 + obstacle.radius
            ):
                coord = self.np_random.random(2) * self.pixel_scale

        return coord

    def _generate_speed(self, speed):
        """生成随机速度(vx, vy)，vx, vy ∈ [-speed, speed]。

        参数：
            speed: 速度范围

        返回：
            (vx, vy): 生成的随机速度
        """
        _speed = (self.np_random.random(2) - 0.5) * 2 * speed

        return _speed[0], _speed[1]

    def add(self):
        """将所有移动对象添加到PyMunk空间。

        创建一个新的PyMunk空间，并将所有对象（追捕者、逃避者、毒物、障碍物）添加到空间中。
        """
        self.space = pymunk.Space()

        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.add(self.space)

    def add_bounding_box(self):
        """创建边界框，防止移动对象逃离视窗。

        四个边界框的排列方式如下：
        (-100, WINDOWSIZE + 100) ┌────┬────────────────────────────┬────┐ (WINDOWSIZE + 100, WINDOWSIZE + 100)
                                 │xxxx│////////////////////////////│xxxx│
                                 ├────┼────────────────────────────┼────┤
                                 │////│    (WINDOWSIZE, WINDOWSIZE)│////│
                                 │////│                            │////│
                                 │////│(0, 0)                      │////│
                                 ├────┼────────────────────────────┼────┤
                                 │xxxx│////////////////////////////│xxxx│
                    (-100, -100) └────┴────────────────────────────┴────┘ (WINDOWSIZE + 100, -100)
        其中"x"表示重叠区域。
        """
        # 边界框边缘
        pts = [
            (-100, -100),
            (self.pixel_scale + 100, -100),
            (self.pixel_scale + 100, self.pixel_scale + 100),
            (-100, self.pixel_scale + 100),
        ]

        self.barriers = []

        for i in range(4):
            self.barriers.append(
                pymunk.Segment(self.space.static_body, pts[i], pts[(i + 1) % 4], 100)
            )
            self.barriers[-1].elasticity = 0.999
            self.space.add(self.barriers[-1])

    def draw(self):
        """在PyGame中绘制所有移动对象和障碍物。

        遍历所有对象列表（追捕者、逃避者、毒物、障碍物），调用每个对象的draw方法进行绘制。
        """
        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.draw(self.screen, self.convert_coordinates)

    def add_handlers(self):
        # 追捕者与逃避者、毒物的碰撞处理器
        self.handlers = []

        for pursuer in self.pursuers:
            for obj in self.evaders:
                self.handlers.append(
                    self.space.add_collision_handler(
                        pursuer.shape.collision_type, obj.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.pursuer_evader_begin_callback
                self.handlers[-1].separate = self.pursuer_evader_separate_callback

            for obj in self.poisons:
                self.handlers.append(
                    self.space.add_collision_handler(
                        pursuer.shape.collision_type, obj.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.pursuer_poison_begin_callback

        # 毒物与逃避者的碰撞处理器
        for poison in self.poisons:
            for evader in self.evaders:
                self.handlers.append(
                    self.space.add_collision_handler(
                        poison.shape.collision_type, evader.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.return_false_begin_callback

        # 逃避者与逃避者的碰撞处理器
        for i in range(self.n_evaders):
            for j in range(i, self.n_evaders):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.evaders[i].shape.collision_type,
                            self.evaders[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

        # 毒物与毒物的碰撞处理器
        for i in range(self.n_poisons):
            for j in range(i, self.n_poisons):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.poisons[i].shape.collision_type,
                            self.poisons[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

        # 追捕者与追捕者的碰撞处理器
        for i in range(self.n_pursuers):
            for j in range(i, self.n_pursuers):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.pursuers[i].shape.collision_type,
                            self.pursuers[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

    def reset(self):
        """重置环境到初始状态。

        返回：
            observations: 所有智能体的初始观察列表
        """
        self.add_obj()
        self.frames = 0

        # 初始化障碍物位置
        if self.initial_obstacle_coord is None:
            for i, obstacle in enumerate(self.obstacles):
                obstacle_position = (
                    self.np_random.random((self.n_obstacles, 2)) * self.pixel_scale
                )
                obstacle.body.position = (
                    obstacle_position[0, 0],
                    obstacle_position[0, 1],
                )
        else:
            for i, obstacle in enumerate(self.obstacles):
                obstacle.body.position = (
                    self.initial_obstacle_coord[i][0] * self.pixel_scale,
                    self.initial_obstacle_coord[i][1] * self.pixel_scale,
                )

        # 将对象添加到空间
        self.add()
        self.add_handlers()
        self.add_bounding_box()

        # 获取观察
        obs_list = self.observe_list()

        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.behavior_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = obs_list

        return obs_list[0]

    def step(self, action, agent_id, is_last):
        """执行一步动作。

        参数：
            action: 要执行的动作
            agent_id: 执行动作的智能体ID
            is_last: 是否是最后一个智能体

        返回：
            observation: 观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        action = np.asarray(action) * self.pursuer_max_accel
        action = action.reshape(2)
        thrust = np.linalg.norm(action)
        if thrust > self.pursuer_max_accel:
            # 限制加速到self.pursuer_max_accel
            action = action * (self.pursuer_max_accel / thrust)

        p = self.pursuers[agent_id]

        # 截断追捕者速度
        _velocity = np.clip(
            p.body.velocity + action * self.pixel_scale,
            -self.pursuer_speed,
            self.pursuer_speed,
        )

        # 设置追捕者速度
        p.reset_velocity(_velocity[0], _velocity[1])

        # 惩罚大动作
        accel_penalty = self.thrust_penalty * math.sqrt((action**2).sum())

        # 平均惩罚在所有代理中，并分配每个代理的全局部分
        self.control_rewards = (
            (accel_penalty / self.n_pursuers)
            * np.ones(self.n_pursuers)
            * (1 - self.local_ratio)
        )

        # 分配当前代理的局部部分
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            self.space.step(1 / self.FPS)

            obs_list = self.observe_list()
            self.last_obs = obs_list

            for id in range(self.n_pursuers):
                p = self.pursuers[id]

                # 奖励为捕获食物粒子、遇到食物粒子和毒物
                self.behavior_rewards[id] = (
                    self.food_reward * p.shape.food_indicator
                    + self.encounter_reward * p.shape.food_touched_indicator
                    + self.poison_reward * p.shape.poison_indicator
                )

                p.shape.food_indicator = 0
                p.shape.poison_indicator = 0

            rewards = np.array(self.behavior_rewards) + np.array(self.control_rewards)

            local_reward = rewards
            global_reward = local_reward.mean()

            # 根据local_ratio分配局部奖励和全局奖励
            self.last_rewards = local_reward * self.local_ratio + global_reward * (
                1 - self.local_ratio
            )

            self.frames += 1

        return self.observe(agent_id)

    def observe(self, agent_id):
        """获取指定智能体的观察。

        参数：
            agent_id: 智能体ID

        返回：
            observation: 智能体的观察
        """
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        """获取所有智能体的观察列表。

        返回：
            observations: 所有智能体的观察列表
        """
        observe_list = []

        for i, pursuer in enumerate(self.pursuers):
            obstacle_distances = []

            evader_distances = []
            evader_velocities = []

            poison_distances = []
            poison_velocities = []

            _pursuer_distances = []
            _pursuer_velocities = []

            for obstacle in self.obstacles:
                obstacle_distance, _ = pursuer.get_sensor_reading(
                    obstacle.body.position, obstacle.radius, obstacle.body.velocity, 0.0
                )
                obstacle_distances.append(obstacle_distance)

            obstacle_sensor_vals = self.get_sensor_readings(
                obstacle_distances, pursuer.sensor_range
            )

            barrier_distances = pursuer.get_sensor_barrier_readings()

            for evader in self.evaders:
                evader_distance, evader_velocity = pursuer.get_sensor_reading(
                    evader.body.position,
                    evader.radius,
                    evader.body.velocity,
                    self.evader_speed,
                )
                evader_distances.append(evader_distance)
                evader_velocities.append(evader_velocity)

            (
                evader_sensor_distance_vals,
                evader_sensor_velocity_vals,
            ) = self.get_sensor_readings(
                evader_distances,
                pursuer.sensor_range,
                velocites=evader_velocities,
            )

            for poison in self.poisons:
                poison_distance, poison_velocity = pursuer.get_sensor_reading(
                    poison.body.position,
                    poison.radius,
                    poison.body.velocity,
                    self.poison_speed,
                )
                poison_distances.append(poison_distance)
                poison_velocities.append(poison_velocity)

            (
                poison_sensor_distance_vals,
                poison_sensor_velocity_vals,
            ) = self.get_sensor_readings(
                poison_distances,
                pursuer.sensor_range,
                velocites=poison_velocities,
            )

            # 当只有一个追捕者时，传感器不会感知其他追捕者
            if self.n_pursuers > 1:
                for j, _pursuer in enumerate(self.pursuers):
                    # 只获取其他追捕者的传感器读数
                    if i == j:
                        continue

                    _pursuer_distance, _pursuer_velocity = pursuer.get_sensor_reading(
                        _pursuer.body.position,
                        _pursuer.radius,
                        _pursuer.body.velocity,
                        self.pursuer_speed,
                    )
                    _pursuer_distances.append(_pursuer_distance)
                    _pursuer_velocities.append(_pursuer_velocity)

                (
                    _pursuer_sensor_distance_vals,
                    _pursuer_sensor_velocity_vals,
                ) = self.get_sensor_readings(
                    _pursuer_distances,
                    pursuer.sensor_range,
                    velocites=_pursuer_velocities,
                )
            else:
                _pursuer_sensor_distance_vals = np.zeros(self.n_sensors)
                _pursuer_sensor_velocity_vals = np.zeros(self.n_sensors)

            if pursuer.shape.food_touched_indicator >= 1:
                food_obs = 1
            else:
                food_obs = 0

            if pursuer.shape.poison_indicator >= 1:
                poison_obs = 1
            else:
                poison_obs = 0

            # 拼接所有观察
            if self.speed_features:
                pursuer_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        evader_sensor_velocity_vals,
                        poison_sensor_distance_vals,
                        poison_sensor_velocity_vals,
                        _pursuer_sensor_distance_vals,
                        _pursuer_sensor_velocity_vals,
                        np.array([food_obs]),
                        np.array([poison_obs]),
                    ]
                )
            else:
                pursuer_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        poison_sensor_distance_vals,
                        _pursuer_sensor_distance_vals,
                        np.array([food_obs]),
                        np.array([poison_obs]),
                    ]
                )

            observe_list.append(pursuer_observation)

        return observe_list

    def get_sensor_readings(self, positions, sensor_range, velocites=None):
        """获取传感器读数。

        参数：
            positions: 所有传感器对所有对象的位置读数
            velocites: 所有传感器对所有对象的速度读数
        """
        distance_vals = np.concatenate(positions, axis=1)

        # 传感器只读取最近的对象
        min_idx = np.argmin(distance_vals, axis=1)

        # 归一化传感器读数
        sensor_distance_vals = np.amin(distance_vals, axis=1)

        if velocites is not None:
            velocity_vals = np.concatenate(velocites, axis=1)

            # 获取最近对象的速度读数
            sensor_velocity_vals = velocity_vals[np.arange(self.n_sensors), min_idx]

            return sensor_distance_vals, sensor_velocity_vals

        return sensor_distance_vals

    def pursuer_poison_begin_callback(self, arbiter, space, data):
        """当追捕者与毒物发生碰撞时调用。

        追捕者的毒物指示器变为1，追捕者在这一步获得惩罚。
        """
        pursuer_shape, poison_shape = arbiter.shapes

        # 给追捕者奖励
        pursuer_shape.poison_indicator += 1

        # 重置毒物位置和速度
        x, y = self._generate_coord(poison_shape.radius)
        vx, vy = self._generate_speed(poison_shape.max_speed)

        poison_shape.reset_position(x, y)
        poison_shape.reset_velocity(vx, vy)

        return False

    def pursuer_evader_begin_callback(self, arbiter, space, data):
        """当追捕者与逃避者发生碰撞时调用。

        逃避者的计数器加1，如果计数器达到n_coop，则追捕者捕获逃避者并获得奖励。
        """
        pursuer_shape, evader_shape = arbiter.shapes

        # 给逃避者添加一次碰撞
        evader_shape.counter += 1

        # 表示食物被追捕者触碰
        pursuer_shape.food_touched_indicator += 1

        if evader_shape.counter >= self.n_coop:
            # 给追捕者奖励
            pursuer_shape.food_indicator = 1

        return False

    def pursuer_evader_separate_callback(self, arbiter, space, data):
        """当追捕者与逃避者结束碰撞时调用。

        如果此时有大于等于n_coop个追捕者与这个逃避者发生碰撞，则重置逃避者的位置，
        并且涉及的追捕者将获得奖励。
        """
        pursuer_shape, evader_shape = arbiter.shapes

        if evader_shape.counter < self.n_coop:
            # 从逃避者移除一次碰撞
            evader_shape.counter -= 1
        else:
            evader_shape.counter = 0

            # 给追捕者奖励
            pursuer_shape.food_indicator = 1

            # 重置逃避者位置和速度
            x, y = self._generate_coord(evader_shape.radius)
            vx, vy = self._generate_speed(evader_shape.max_speed)

            evader_shape.reset_position(x, y)
            evader_shape.reset_velocity(vx, vy)

        pursuer_shape.food_touched_indicator -= 1

    def return_false_begin_callback(self, arbiter, space, data):
        """简单返回False的回调函数。"""
        return False

    def render(self):
        """渲染环境。

        如果render_mode为human，则使用pygame窗口显示；
        如果render_mode为rgb_array，则返回RGB数组。
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "你正在调用render方法但没有指定任何渲染模式。"
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale, self.pixel_scale)
                )
                pygame.display.set_caption("水世界")
            else:
                self.screen = pygame.Surface((self.pixel_scale, self.pixel_scale))

        self.screen.fill((255, 255, 255))
        self.draw()
        self.clock.tick(self.FPS)

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
