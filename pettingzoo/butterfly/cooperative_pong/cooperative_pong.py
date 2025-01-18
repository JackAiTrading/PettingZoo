"""
合作乒乓球环境。

这个环境模拟了一个多人合作的乒乓球游戏，玩家控制球拍
共同配合来保持球的运动，通过团队合作获得高分。

主要特点：
1. 多人合作
2. 物理模拟
3. 策略配合
4. 实时控制

环境规则：
1. 基本设置
   - 球拍位置
   - 球的物理
   - 得分规则
   - 边界处理

2. 交互规则
   - 移动控制
   - 碰撞反弹
   - 速度变化
   - 分数计算

3. 智能体行为
   - 位置预判
   - 速度控制
   - 角度调整
   - 团队配合

4. 终止条件
   - 球出界
   - 达到分数
   - 时间耗尽
   - 任务完成

环境参数：
- 观察空间：游戏场景状态
- 动作空间：球拍移动控制
- 奖励：保球和配合得分
- 最大步数：由时间限制决定

环境特色：
1. 物理系统
   - 球的运动
   - 碰撞检测
   - 速度计算
   - 角度反射

2. 控制机制
   - 球拍移动
   - 击球反弹
   - 速度调节
   - 位置限制

3. 策略元素
   - 位置选择
   - 速度控制
   - 角度预判
   - 团队协作

4. 评估系统
   - 保球时间
   - 配合次数
   - 团队表现
   - 整体得分

注意事项：
- 位置预判
- 速度控制
- 团队配合
- 策略调整
"""

"""
合作乒乓球游戏环境。

这个环境实现了一个双人合作的乒乓球游戏，两个玩家需要通过合作来尽可能长时间地保持球在场内。

主要特点：
1. 双人合作模式
2. 连续的动作空间
3. 基于物理的球体运动
4. 支持手动和AI控制

游戏规则：
1. 两名玩家各控制一个球拍
2. 球拍可以上下移动来击打球
3. 当球触及左右边界时游戏结束
4. 玩家需要合作尽可能长时间保持球在场内

环境参数：
- 观察空间：RGB图像 (280, 240, 3)
- 动作空间：连续值 [-1, 1]，控制球拍上下移动
- 奖励：球保持在场内时为正，出界时为负
"""

"""
合作乒乓球游戏环境。

这个模块实现了一个双人合作的乒乓球游戏，两名玩家需要协同配合来保持球在游戏中。
游戏包含合作元素，玩家需要互相配合以获得更高的分数。

这个环境是蝴蝶环境的一部分。请先阅读该页面的通用信息。

| 导入             | `from pettingzoo.butterfly import cooperative_pong_v5` |
|--------------------|---------------------------------------------------|
| 动作类型          | 离散                                              |
| 并行API          | 是                                                |
| 手动控制         | 是                                                |
| 智能体           | `agents= ['paddle_0', 'paddle_1']`                |
| 智能体数量       | 2                                                 |
| 动作形状         | Discrete(3)                                       |
| 动作值           | [0, 2]                                            |
| 观察形状         | (280, 480, 3)                                     |
| 观察值           | (0, 255)                                          |
| 平均总奖励       | 0.91                                              |

## 描述

合作乒乓球是一个双人合作游戏，玩家需要控制挡板来击打球。每当球被击中时，玩家获得1分。
当球碰到左边或右边的墙壁时，游戏结束。

### 观察空间

观察空间是一个 RGB 图像，大小为 280x480x3。这代表了游戏的当前状态，包括球和挡板的位置。

### 动作空间

每个智能体有 3 种可能的动作：

| 动作编号 | 动作     |
|----------|----------|
| 0        | 不动作   |
| 1        | 向上移动 |
| 2        | 向下移动 |

### 奖励

每个智能体在每一步都会获得以下奖励之一：
* 1：成功击打球
* 0：其他情况
* -1：球碰到左边或右边的墙壁（游戏结束）

### 版本历史

* v5: 修复了渲染 (1.14.0)
* v4: 修复了渲染 (1.9.0)
* v3: 修复了渲染 (1.8.2)
* v2: 修复了渲染 (1.4.0)
* v1: 修复了观察空间和文档 (1.4.0)
* v0: 初始版本发布 (1.0.0)
"""

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv

from .paddle import Paddle


def env(render_mode=None):
    """创建合作乒乓球游戏环境的包装器。

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
    """合作乒乓球游戏的主要环境类。

    这个环境实现了一个双人合作的乒乓球游戏，玩家需要协同配合来保持球在游戏中。

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
        "name": "cooperative_pong_v5",
        "is_parallelizable": True,
        "render_fps": 15,
    }

    def __init__(self, render_mode=None):
        """初始化合作乒乓球游戏环境。

        参数:
            render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"
        """
        super().__init__()
        EzPickle.__init__(self, render_mode)

        self.screen_width = 480
        self.screen_height = 280
        self.render_mode = render_mode

        # 初始化 pygame
        pygame.init()
        self.clock = pygame.time.Clock()

        # 定义智能体
        self.possible_agents = ["paddle_0", "paddle_1"]
        self.agents = self.possible_agents[:]

        # 定义动作和观察空间
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=255,
                shape=(self.screen_height, self.screen_width, 3),
                dtype=np.uint8,
            )
            for agent in self.possible_agents
        }

        # 游戏对象
        self.paddles = {}
        self.ball_pos = None
        self.ball_vel = None

        # 渲染相关
        self.render_surface = pygame.Surface((self.screen_width, self.screen_height))
        if self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Cooperative Pong")
            self.window_surface = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )

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
            spaces.Discrete: 动作空间
        """
        return self.action_spaces[agent]

    def observe(self, agent):
        """获取指定智能体的观察。

        参数:
            agent (str): 智能体标识符

        返回:
            np.ndarray: RGB图像形式的观察
        """
        return np.copy(pygame.surfarray.array3d(self.render_surface).swapaxes(0, 1))

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

        # 初始化挡板
        self.paddles = {
            "paddle_0": Paddle(20, self.screen_height // 2),
            "paddle_1": Paddle(self.screen_width - 20, self.screen_height // 2),
        }

        # 初始化球
        self.ball_pos = np.array([self.screen_width // 2, self.screen_height // 2])
        self.ball_vel = np.array([4.0 * (1.0 if self.np_random.random() < 0.5 else -1.0),
                                 4.0 * (1.0 if self.np_random.random() < 0.5 else -1.0)])

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
            action (int): 要执行的动作

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

        # 更新挡板位置
        if action == 1:  # 向上移动
            self.paddles[agent].move_up()
        elif action == 2:  # 向下移动
            self.paddles[agent].move_down()

        # 如果是最后一个智能体，更新球的位置
        if self._agent_selector.is_last():
            # 更新球的位置
            self.ball_pos += self.ball_vel

            # 检查与挡板的碰撞
            for paddle in self.paddles.values():
                if paddle.check_collision(self.ball_pos):
                    self.ball_vel[0] *= -1
                    self.rewards = {agent: 1 for agent in self.agents}

            # 检查与上下墙壁的碰撞
            if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.screen_height:
                self.ball_vel[1] *= -1

            # 检查与左右墙壁的碰撞（游戏结束条件）
            if self.ball_pos[0] <= 0 or self.ball_pos[0] >= self.screen_width:
                self.rewards = {agent: -1 for agent in self.agents}
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
            return np.copy(pygame.surfarray.array3d(self.render_surface).swapaxes(0, 1))

    def close(self):
        """关闭环境，释放资源。"""
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.quit()

    def _render_frame(self):
        """渲染一帧游戏画面。"""
        # 清空屏幕
        self.render_surface.fill((0, 0, 0))

        # 绘制挡板
        for paddle in self.paddles.values():
            pygame.draw.rect(
                self.render_surface,
                (255, 255, 255),
                pygame.Rect(
                    paddle.x - 5,
                    paddle.y - 20,
                    10,
                    40
                )
            )

        # 绘制球
        pygame.draw.circle(
            self.render_surface,
            (255, 255, 255),
            self.ball_pos.astype(int),
            5
        )

        if self.render_mode == "human":
            self.window_surface.blit(self.render_surface, (0, 0))
