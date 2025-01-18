"""
活塞球环境。

这个环境模拟了一个多人合作的物理游戏，玩家控制活塞
共同配合来移动球体到目标位置，通过团队合作获得高分。

主要特点：
1. 多人合作
2. 物理模拟
3. 策略配合
4. 实时控制

环境规则：
1. 基本设置
   - 活塞位置
   - 球的物理
   - 得分规则
   - 边界处理

2. 交互规则
   - 活塞控制
   - 碰撞反弹
   - 速度变化
   - 分数计算

3. 智能体行为
   - 位置预判
   - 力度控制
   - 时机把握
   - 团队配合

4. 终止条件
   - 球到达
   - 时间耗尽
   - 达到分数
   - 任务完成

环境参数：
- 观察空间：游戏场景状态
- 动作空间：活塞控制
- 奖励：球的前进和配合
- 最大步数：由时间限制决定

环境特色：
1. 物理系统
   - 球的运动
   - 碰撞检测
   - 速度计算
   - 力的传递

2. 控制机制
   - 活塞移动
   - 撞击反弹
   - 速度调节
   - 位置限制

3. 策略元素
   - 位置选择
   - 力度控制
   - 时机把握
   - 团队协作

4. 评估系统
   - 前进距离
   - 配合次数
   - 团队表现
   - 整体得分

注意事项：
- 位置预判
- 力度控制
- 团队配合
- 策略调整
"""

"""
活塞球游戏环境模块。

这个模块实现了一个多智能体合作游戏环境，其中多个活塞需要协同工作，
将一个球推向右侧以获得奖励。

主要功能：
1. 实现活塞球游戏的核心逻辑
2. 提供多智能体协作环境
3. 支持自定义渲染和观察空间

游戏规则：
1. 多个活塞（智能体）需要协同工作
2. 目标是将球推向右侧
3. 每个活塞可以上下移动
4. 当球到达右侧时获得正奖励
"""

import math
import os
from typing import Dict, Optional, Union

import gymnasium
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding

from pettingzoo.utils.env import ParallelEnv

# 游戏窗口尺寸
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 560

# 物理参数
FPS = 50
PHYSICS_STEPS = 1
GRAVITY = 0.0
FRICTION = 0.7
DAMPING = 1.0
BALL_DENSITY = 1.0
BALL_RADIUS = 40
PISTON_WIDTH = 80
PISTON_HEIGHT = 20
PISTON_GAP = 10
MAX_PISTON_SPEED = 30


class raw_env(ParallelEnv, EzPickle):
    """
    活塞球游戏环境类。

    这个类实现了活塞球游戏的核心逻辑，包括物理模拟、
    状态管理和智能体交互。

    属性：
        agents (list): 所有智能体的列表
        possible_agents (list): 所有可能的智能体列表
        action_spaces (dict): 每个智能体的动作空间
        observation_spaces (dict): 每个智能体的观察空间
        metadata (dict): 环境元数据
        render_mode (str): 渲染模式
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pistonball_v6",
        "is_parallelizable": True,
        "render_fps": FPS,
    }

    def __init__(
        self,
        n_pistons: int = 15,
        time_penalty: float = 0.0,
        continuous: bool = False,
        random_drop: bool = True,
        random_rotate: bool = True,
        ball_mass: float = 0.75,
        ball_friction: float = 0.3,
        ball_elasticity: float = 1.5,
        max_cycles: int = 900,
        render_mode: Optional[str] = None,
    ):
        """
        初始化活塞球游戏环境。

        参数：
            n_pistons (int): 活塞数量
            time_penalty (float): 时间惩罚系数
            continuous (bool): 是否使用连续动作空间
            random_drop (bool): 是否随机放置球
            random_rotate (bool): 是否随机旋转球
            ball_mass (float): 球的质量
            ball_friction (float): 球的摩擦系数
            ball_elasticity (float): 球的弹性系数
            max_cycles (int): 最大步数
            render_mode (str): 渲染模式
        """
        EzPickle.__init__(
            self,
            n_pistons,
            time_penalty,
            continuous,
            random_drop,
            random_rotate,
            ball_mass,
            ball_friction,
            ball_elasticity,
            max_cycles,
            render_mode,
        )

        # 初始化参数
        self.n_pistons = n_pistons
        self.time_penalty = time_penalty
        self.continuous = continuous
        self.random_drop = random_drop
        self.random_rotate = random_rotate
        self.ball_mass = ball_mass
        self.ball_friction = ball_friction
        self.ball_elasticity = ball_elasticity
        self.max_cycles = max_cycles
        self.render_mode = render_mode

        # 初始化智能体
        self.possible_agents = [f"piston_{i}" for i in range(self.n_pistons)]
        self.agents = self.possible_agents[:]

        # 设置动作空间
        if self.continuous:
            self.action_spaces = {
                agent: spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                for agent in self.possible_agents
            }
        else:
            self.action_spaces = {
                agent: spaces.Discrete(3) for agent in self.possible_agents
            }

        # 设置观察空间
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=255, shape=(100, 120, 3), dtype=np.uint8
            )
            for agent in self.possible_agents
        }

        # 初始化物理引擎
        self.screen = None
        self.clock = None
        self.space = None
        self.ball = None
        self.pistons = []
        self.walls = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[Dict, Dict]:
        """
        重置环境到初始状态。

        参数：
            seed (int): 随机数种子
            options (dict): 重置选项

        返回：
            tuple: (observations, infos)
                - observations: 每个智能体的初始观察
                - infos: 每个智能体的初始信息
        """
        # 重置随机数生成器
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # 重置智能体列表
        self.agents = self.possible_agents[:]

        # 重置物理引擎
        self.space = pymunk.Space()
        self.space.gravity = (0.0, GRAVITY)
        self.space.damping = DAMPING

        # 创建墙壁
        self._create_walls()

        # 创建活塞
        self._create_pistons()

        # 创建球
        self._create_ball()

        # 获取初始观察
        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(
        self, actions: Dict
    ) -> tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        执行一步环境交互。

        参数：
            actions (Dict): 每个智能体的动作

        返回：
            tuple: (observations, rewards, terminations, truncations, infos)
                - observations: 每个智能体的新观察
                - rewards: 每个智能体的奖励
                - terminations: 每个智能体的终止状态
                - truncations: 每个智能体的截断状态
                - infos: 每个智能体的额外信息
        """
        # 处理每个智能体的动作
        for agent_id, action in actions.items():
            piston = self.pistons[int(agent_id.split("_")[1])]
            if self.continuous:
                piston.velocity = (0, action[0] * MAX_PISTON_SPEED)
            else:
                if action == 1:
                    piston.velocity = (0, MAX_PISTON_SPEED)
                elif action == 2:
                    piston.velocity = (0, -MAX_PISTON_SPEED)
                else:
                    piston.velocity = (0, 0)

        # 更新物理引擎
        for _ in range(PHYSICS_STEPS):
            self.space.step(1.0 / FPS)

        # 计算奖励
        reward = self._get_reward()
        rewards = {agent: reward for agent in self.agents}

        # 检查是否结束
        done = self._is_done()
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # 获取观察和信息
        observations = {agent: self._get_obs() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        渲染环境的当前状态。

        返回：
            Optional[Union[np.ndarray, str]]: 渲染结果
                - 如果render_mode为"human"，显示窗口并返回None
                - 如果render_mode为"rgb_array"，返回RGB数组
        """
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()

        # 清空屏幕
        self.screen.fill((255, 255, 255))

        # 绘制所有对象
        options = pymunk.pygame_util.DrawOptions(self.screen)
        self.space.debug_draw(options)

        # 更新显示
        pygame.display.flip()

        # 控制帧率
        self.clock.tick(FPS)

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """
        关闭环境，释放资源。
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _create_walls(self):
        """
        创建游戏边界墙壁。
        """
        # 创建四周的墙壁
        walls = [
            [(0, 0), (0, WINDOW_HEIGHT)],  # 左墙
            [(0, 0), (WINDOW_WIDTH, 0)],  # 上墙
            [(WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT)],  # 右墙
            [(0, WINDOW_HEIGHT), (WINDOW_WIDTH, WINDOW_HEIGHT)],  # 下墙
        ]

        for wall in walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, wall[0], wall[1], 1.0)
            shape.friction = FRICTION
            self.space.add(body, shape)
            self.walls.append(shape)

    def _create_pistons(self):
        """
        创建活塞。
        """
        for i in range(self.n_pistons):
            x = PISTON_GAP + i * (PISTON_WIDTH + PISTON_GAP)
            y = WINDOW_HEIGHT // 2

            # 创建活塞主体
            body = pymunk.Body(1.0, float("inf"))
            body.position = (x, y)

            # 创建活塞形状
            shape = pymunk.Poly.create_box(body, (PISTON_WIDTH, PISTON_HEIGHT))
            shape.friction = FRICTION

            # 添加到物理引擎
            self.space.add(body, shape)
            self.pistons.append(body)

    def _create_ball(self):
        """
        创建球。
        """
        mass = self.ball_mass
        radius = BALL_RADIUS
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))

        # 创建球体
        body = pymunk.Body(mass, inertia)
        x = WINDOW_WIDTH // 4
        if self.random_drop:
            x = self._np_random.uniform(WINDOW_WIDTH // 4, WINDOW_WIDTH // 2)
        y = WINDOW_HEIGHT // 2
        body.position = (x, y)

        if self.random_rotate:
            body.angle = self._np_random.uniform(0, 2 * math.pi)

        # 创建球形状
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.friction = self.ball_friction
        shape.elasticity = self.ball_elasticity

        # 添加到物理引擎
        self.space.add(body, shape)
        self.ball = body

    def _get_obs(self) -> np.ndarray:
        """
        获取当前状态的观察。

        返回：
            np.ndarray: 游戏画面的RGB数组
        """
        # 渲染当前状态
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        # 清空屏幕并绘制
        self.screen.fill((255, 255, 255))
        options = pymunk.pygame_util.DrawOptions(self.screen)
        self.space.debug_draw(options)

        # 获取画面数组
        observation = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

        return observation

    def _get_reward(self) -> float:
        """
        计算当前状态的奖励。

        返回：
            float: 奖励值
        """
        # 基础奖励：球的x坐标进展
        reward = self.ball.position[0] / WINDOW_WIDTH

        # 时间惩罚
        if self.time_penalty:
            reward -= self.time_penalty

        return reward

    def _is_done(self) -> bool:
        """
        检查游戏是否结束。

        返回：
            bool: 如果游戏结束返回True，否则返回False
        """
        # 检查球是否到达右边界
        if self.ball.position[0] >= WINDOW_WIDTH:
            return True

        # 检查是否超过最大步数
        if self.max_cycles and self.max_cycles <= 0:
            return True

        return False
