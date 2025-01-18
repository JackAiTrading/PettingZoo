"""
活塞球游戏环境。

这个模块实现了一个多智能体的物理引擎游戏，多个智能体控制活塞来移动球体到目标位置。
游戏包含合作元素，智能体需要协同配合来获得更高分数。

这个环境是蝴蝶环境的一部分。请先阅读该页面的通用信息。

| 导入             | `from pettingzoo.butterfly import pistonball_v6` |
|--------------------|------------------------------------------------|
| 动作类型          | 离散                                           |
| 并行API          | 是                                             |
| 手动控制         | 是                                             |
| 智能体           | `agents= ['piston_0', 'piston_1', ..., 'piston_19']` |
| 智能体数量       | 20                                             |
| 动作形状         | Discrete(3)                                    |
| 动作值           | [0, 2]                                         |
| 观察形状         | (200, 120, 3)                                  |
| 观察值           | (0, 255)                                       |
| 平均总奖励       | 30.0                                           |

## 描述

活塞球是一个物理引擎游戏，其中20个智能体控制活塞来移动一个球体。每个智能体可以控制一个活塞上下移动。
智能体的目标是将球移动到屏幕右侧，同时防止球掉落到地面。球每向右移动一个单位就会获得奖励，如果球掉落到地面则获得负奖励。

### 观察空间

观察空间是一个 RGB 图像，大小为 200x120x3。这代表了游戏的当前状态，包括球和活塞的位置。

### 动作空间

每个智能体有 3 种可能的动作：

| 动作编号 | 动作     |
|----------|----------|
| 0        | 不动作   |
| 1        | 向上移动 |
| 2        | 向下移动 |

### 奖励

每个智能体在每一步都会获得以下奖励：
* +1：球向右移动一个单位
* -1：球向左移动一个单位
* -2：球掉落到地面

### 版本历史

* v6: 修复了渲染 (1.14.0)
* v5: 修复了渲染 (1.9.0)
* v4: 修复了渲染 (1.8.2)
* v3: 修复了渲染 (1.4.0)
* v2: 修复了观察空间和文档 (1.4.0)
* v1: 修复了渲染 (1.3.1)
* v0: 初始版本发布 (1.0.0)
"""

import gymnasium
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv


def env(render_mode=None):
    """创建活塞球游戏环境的包装器。

    参数:
        render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"

    返回:
        AECEnv: 包装后的环境
    """
    env = raw_env(render_mode=render_mode)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    """活塞球游戏的主要环境类。

    这个环境实现了一个多智能体的物理引擎游戏，智能体通过控制活塞来移动球体。

    属性:
        metadata (dict): 环境的元数据，包括版本信息和渲染模式
        possible_agents (list): 可能的智能体列表
        screen_width (int): 屏幕宽度
        screen_height (int): 屏幕高度
        action_spaces (dict): 每个智能体的动作空间
        observation_spaces (dict): 每个智能体的观察空间
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pistonball_v6",
        "is_parallelizable": True,
        "render_fps": 50,
    }

    def __init__(self, render_mode=None, n_pistons=20, time_penalty=-0.1, continuous=False, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900):
        """初始化活塞球游戏环境。

        参数:
            render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"
            n_pistons (int, 可选): 活塞数量，默认为20
            time_penalty (float, 可选): 每步的时间惩罚，默认为-0.1
            continuous (bool, 可选): 是否使用连续动作空间，默认为False
            random_drop (bool, 可选): 是否随机初始化球的位置，默认为True
            random_rotate (bool, 可选): 是否随机初始化球的旋转，默认为True
            ball_mass (float, 可选): 球的质量，默认为0.75
            ball_friction (float, 可选): 球的摩擦系数，默认为0.3
            ball_elasticity (float, 可选): 球的弹性系数，默认为1.5
            max_cycles (int, 可选): 最大游戏周期数，默认为900
        """
        super().__init__()
        EzPickle.__init__(
            self,
            render_mode,
            n_pistons,
            time_penalty,
            continuous,
            random_drop,
            random_rotate,
            ball_mass,
            ball_friction,
            ball_elasticity,
            max_cycles,
        )

        self.n_pistons = n_pistons
        self.time_penalty = time_penalty
        self.continuous = continuous
        self.random_drop = random_drop
        self.random_rotate = random_rotate
        self.ball_mass = ball_mass
        self.ball_friction = ball_friction
        self.ball_elasticity = ball_elasticity
        self.max_cycles = max_cycles

        self.screen_width = 960
        self.screen_height = 560
        self.render_mode = render_mode

        # 初始化 pygame 和 pymunk
        pygame.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # 设置物理引擎
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 750.0)

        # 定义智能体
        self.possible_agents = [f"piston_{i}" for i in range(n_pistons)]
        self.agents = self.possible_agents[:]

        # 定义动作和观察空间
        if continuous:
            self.action_spaces = {
                agent: spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                for agent in self.possible_agents
            }
        else:
            self.action_spaces = {
                agent: spaces.Discrete(3) for agent in self.possible_agents
            }

        self.observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=255,
                shape=(200, 120, 3),
                dtype=np.uint8,
            )
            for agent in self.possible_agents
        }

        # 游戏对象
        self.pistons = []
        self.ball = None

        # 渲染相关
        if self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Pistonball")
            self.window_surface = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def observation_space(self, agent):
        """获取指定智能体的观察空间。

        参数:
            agent (str): 智能体标识符

        返回:
            spaces.Box: 观察空间
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """获取指定智能体的动作空间。

        参数:
            agent (str): 智能体标识符

        返回:
            spaces.Box 或 spaces.Discrete: 动作空间
        """
        return self.action_spaces[agent]

    def observe(self, agent):
        """获取指定智能体的观察。

        参数:
            agent (str): 智能体标识符

        返回:
            np.ndarray: RGB图像形式的观察
        """
        observation = pygame.surfarray.array3d(self.screen)
        observation = np.transpose(observation, axes=(1, 0, 2))
        return observation

    def reset(self, seed=None, options=None):
        """重置游戏到初始状态。

        参数:
            seed (int, 可选): 随机种子
            options (dict, 可选): 重置选项

        返回:
            observations (dict): 每个智能体的初始观察
            infos (dict): 每个智能体的初始信息
        """
        super().reset(seed=seed)

        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # 清空物理引擎
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 750.0)

        # 创建边界
        self._create_walls()

        # 创建活塞
        self.pistons = []
        for i in range(self.n_pistons):
            piston = self._create_piston(i)
            self.pistons.append(piston)

        # 创建球
        self.ball = self._create_ball()

        # 初始化奖励和状态
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # 渲染初始状态
        self._render_frame()

        observations = {agent: self.observe(agent) for agent in self.agents}

        return observations, self.infos

    def step(self, action):
        """执行一步动作。

        参数:
            action (float 或 int): 要执行的动作

        返回:
            observations (dict): 每个智能体的新观察
            rewards (dict): 每个智能体的奖励
            terminations (dict): 每个智能体的终止状态
            truncations (dict): 每个智能体的截断状态
            infos (dict): 每个智能体的信息
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        agent_id = int(agent.split("_")[1])

        # 更新活塞位置
        if self.continuous:
            force = action[0] * 200
        else:
            force = {0: 0, 1: -200, 2: 200}[action]

        self.pistons[agent_id].body.apply_force_at_local_point((0, force), (0, 0))

        # 如果是最后一个智能体，更新物理引擎
        if self._agent_selector.is_last():
            # 更新物理引擎
            self.space.step(1.0 / 50.0)

            # 计算奖励
            reward = self._get_reward()
            self.rewards = {agent: reward for agent in self.agents}

            # 检查游戏是否结束
            if self.ball.body.position.y > self.screen_height:
                self.terminations = {agent: True for agent in self.agents}

            # 渲染新的状态
            self._render_frame()

        # 更新智能体选择
        self.agent_selection = self._agent_selector.next()

        observations = {agent: self.observe(agent) for agent in self.agents}

        return (
            observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def render(self):
        """渲染当前游戏状态。

        返回:
            pygame.Surface 或 np.ndarray: 根据渲染模式返回游戏画面
        """
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return self.window_surface
        else:
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                axes=(1, 0, 2)
            )

    def close(self):
        """关闭环境，释放资源。"""
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.quit()

    def _create_walls(self):
        """创建游戏边界墙壁。"""
        static_body = self.space.static_body

        # 创建四面墙
        walls = [
            pymunk.Segment(static_body, (0, 0), (0, self.screen_height), 1),
            pymunk.Segment(static_body, (0, self.screen_height), (self.screen_width, self.screen_height), 1),
            pymunk.Segment(static_body, (self.screen_width, self.screen_height), (self.screen_width, 0), 1),
            pymunk.Segment(static_body, (self.screen_width, 0), (0, 0), 1),
        ]

        for wall in walls:
            wall.friction = 0.5
            wall.elasticity = 0.95
            self.space.add(wall)

    def _create_piston(self, i):
        """创建一个活塞。

        参数:
            i (int): 活塞的索引

        返回:
            pymunk.Body: 活塞的物理对象
        """
        x = self.screen_width * (i + 0.5) / self.n_pistons
        y = self.screen_height - 80

        piston = pymunk.Body(1.0, float("inf"))
        piston.position = (x, y)

        segment = pymunk.Segment(piston, (0, 0), (0, 40), 5)
        segment.friction = 0.5
        segment.elasticity = 0.95

        joint = pymunk.PinJoint(self.space.static_body, piston, (x, y), (0, 0))
        
        self.space.add(piston, segment, joint)
        return piston

    def _create_ball(self):
        """创建球体。

        返回:
            pymunk.Body: 球的物理对象
        """
        mass = self.ball_mass
        radius = 40
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        
        if self.random_drop:
            x = self.np_random.uniform(radius, self.screen_width - radius)
            y = self.np_random.uniform(120, 400)
        else:
            x = self.screen_width / 4
            y = self.screen_height / 2

        body.position = (x, y)
        
        if self.random_rotate:
            body.angle = self.np_random.uniform(0, 2 * np.pi)

        shape = pymunk.Circle(body, radius)
        shape.friction = self.ball_friction
        shape.elasticity = self.ball_elasticity

        self.space.add(body, shape)
        return shape

    def _get_reward(self):
        """计算奖励。

        返回:
            float: 奖励值
        """
        reward = 0.0

        # 根据球的位置计算奖励
        x = self.ball.body.position.x
        if x > self.last_x:
            reward += 1.0
        elif x < self.last_x:
            reward -= 1.0

        # 如果球掉落到地面
        if self.ball.body.position.y > self.screen_height:
            reward -= 2.0

        # 添加时间惩罚
        reward += self.time_penalty

        self.last_x = x
        return reward

    def _render_frame(self):
        """渲染一帧游戏画面。"""
        # 清空屏幕
        self.screen.fill((0, 0, 0))

        # 绘制所有物理对象
        self.space.debug_draw(self.draw_options)

        if self.render_mode == "human":
            self.window_surface.blit(self.screen, (0, 0))
