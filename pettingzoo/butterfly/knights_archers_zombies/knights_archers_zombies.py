"""
骑士射手大战僵尸环境。

这个环境模拟了一个多人合作的动作游戏，玩家控制骑士和射手
共同对抗不断涌来的僵尸，通过团队配合来生存并获得高分。

主要特点：
1. 多人合作
2. 动作对抗
3. 资源管理
4. 团队配合

环境规则：
1. 基本设置
   - 角色类型
   - 武器系统
   - 生命值
   - 得分规则

2. 交互规则
   - 移动控制
   - 攻击方式
   - 碰撞检测
   - 伤害计算

3. 智能体行为
   - 目标选择
   - 战术配合
   - 位置调整
   - 资源利用

4. 终止条件
   - 全员阵亡
   - 达到时限
   - 达到分数
   - 任务完成

环境参数：
- 观察空间：游戏场景状态
- 动作空间：移动和攻击选择
- 奖励：击杀得分和生存奖励
- 最大步数：由时间限制决定

环境特色：
1. 角色系统
   - 骑士特性
   - 射手特性
   - 僵尸特性
   - 武器属性

2. 战斗机制
   - 攻击方式
   - 伤害计算
   - 碰撞检测
   - 生命恢复

3. 策略元素
   - 位置选择
   - 目标优先级
   - 资源管理
   - 团队协作

4. 评估系统
   - 击杀数量
   - 生存时间
   - 团队配合
   - 整体表现

注意事项：
- 位置选择
- 目标优先级
- 团队配合
- 资源管理
"""

"""
骑士弓箭手大战僵尸游戏环境。

这个模块实现了一个多智能体游戏，玩家控制骑士和弓箭手对抗僵尸。
游戏包含合作元素，玩家需要协同作战以生存并击败僵尸。

这个环境是蝴蝶环境的一部分。请先阅读该页面的通用信息。

| 导入             | `from pettingzoo.butterfly import knights_archers_zombies_v10` |
|--------------------|----------------------------------------------------------|
| 动作类型          | 离散                                                     |
| 并行API          | 是                                                       |
| 手动控制         | 是                                                       |
| 智能体           | `agents= ['knight_0', 'archer_0', 'knight_1', 'archer_1']` |
| 智能体数量       | 4                                                        |
| 动作形状         | Discrete(6)                                              |
| 动作值           | [0, 5]                                                   |
| 观察形状         | (512, 512, 3)                                           |
| 观察值           | (0, 255)                                                |
| 平均总奖励       | -5.3                                                    |

## 描述

骑士弓箭手大战僵尸是一个团队生存游戏，玩家需要尽可能长时间地生存。游戏中有两种类型的玩家：骑士和弓箭手。骑士使用近战武器（剑），而弓箭手使用远程武器（弓箭）。

僵尸会从地图的四周生成并向玩家移动。如果僵尸碰到玩家，玩家就会死亡。玩家可以通过攻击来消灭僵尸。当所有玩家都死亡时游戏结束。

### 观察空间

观察空间是一个 RGB 图像，大小为 512x512x3。这代表了游戏的当前状态，包括所有玩家、僵尸和地图元素的位置。

### 动作空间

智能体有 6 种可能的动作：

| 动作编号 | 动作     |
|----------|----------|
| 0        | 不动作   |
| 1        | 向右移动 |
| 2        | 向左移动 |
| 3        | 向上移动 |
| 4        | 向下移动 |
| 5        | 攻击     |

### 奖励

每个智能体在每一步都会获得以下奖励之一：
* -1：每一步的时间惩罚
* -50：死亡惩罚
* 5：击杀僵尸奖励

### 版本历史

* v10: 修复了渲染 (1.14.0)
* v9: 修复了渲染 (1.11.0)
* v8: 修复了渲染 (1.9.0)
* v7: 修复了渲染 (1.8.2)
* v6: 修复了渲染 (1.8.0)
* v5: 修复了渲染 (1.7.0)
* v4: 修复了渲染 (1.6.0)
* v3: 修复了渲染 (1.5.0)
* v2: 修复了渲染 (1.4.0)
* v1: 修复了渲染 (1.3.1)
* v0: 初始版本发布 (1.0.0)
"""

import os
import sys

import gymnasium
import numpy as np
import pygame
import pygame.gfxdraw
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv

from .src import constants as const
from .src import variables as var
from .src.img import get_image
from .src.players import Archer, Knight
from .src.weapons import Arrow, Sword
from .src.zombie import Zombie


def env(render_mode=None, **kwargs):
    """创建骑士弓箭手大战僵尸游戏环境的包装器。

    参数:
        render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"
        **kwargs: 其他参数

    返回:
        AECEnv: 包装后的环境
    """
    env = raw_env(render_mode=render_mode, **kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    """骑士弓箭手大战僵尸游戏的主要环境类。

    这个环境实现了一个多智能体游戏，玩家控制骑士和弓箭手对抗僵尸。

    属性:
        metadata (dict): 环境的元数据，包括版本信息和渲染模式
        possible_agents (list): 可能的智能体列表
        action_spaces (dict): 每个智能体的动作空间
        observation_spaces (dict): 每个智能体的观察空间
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "knights_archers_zombies_v10",
        "is_parallelizable": True,
        "render_fps": const.FPS,
    }

    def __init__(self, render_mode=None, spawn_rate=20, num_archers=2, num_knights=2, max_zombies=10, max_arrows=100):
        """初始化骑士弓箭手大战僵尸游戏环境。

        参数:
            render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"
            spawn_rate (int, 可选): 僵尸生成速率，默认为 20
            num_archers (int, 可选): 弓箭手数量，默认为 2
            num_knights (int, 可选): 骑士数量，默认为 2
            max_zombies (int, 可选): 最大僵尸数量，默认为 10
            max_arrows (int, 可选): 最大箭矢数量，默认为 100
        """
        super().__init__()
        EzPickle.__init__(
            self,
            render_mode,
            spawn_rate,
            num_archers,
            num_knights,
            max_zombies,
            max_arrows,
        )

        self.render_mode = render_mode
        self.spawn_rate = spawn_rate
        self.num_archers = num_archers
        self.num_knights = num_knights
        self.max_zombies = max_zombies
        self.max_arrows = max_arrows

        pygame.init()
        self.screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        self.render_surface = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))

        self.agents = []
        for i in range(self.num_knights):  # 添加骑士
            self.agents.append(f"knight_{i}")
        for i in range(self.num_archers):  # 添加弓箭手
            self.agents.append(f"archer_{i}")
        self.possible_agents = self.agents[:]

        self.action_spaces = {
            name: spaces.Discrete(6) for name in self.possible_agents
        }
        self.observation_spaces = {
            name: spaces.Box(
                low=0,
                high=255,
                shape=(const.SCREEN_WIDTH, const.SCREEN_HEIGHT, 3),
                dtype=np.uint8,
            )
            for name in self.possible_agents
        }

        if self.render_mode == "human":
            pygame.display.init()
            self.window_surface = pygame.display.set_mode(
                (const.SCREEN_WIDTH, const.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Knights, Archers, Zombies")

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
        if agent not in self.agents:
            return None

        pygame.surfarray.array3d(self.render_surface).swapaxes(0, 1)

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

        # 初始化游戏组件
        self.game_over = False
        self.spawn_rate = self.spawn_rate
        self.knight_player_num = self.num_knights
        self.archer_player_num = self.num_archers
        self.score = 0
        self.run_time = 0
        self.zombie_spawn_rate = 0
        self.knight_dict = {}
        self.archer_dict = {}
        self.knight_list = []
        self.archer_list = []
        self.arrow_list = []
        self.zombie_list = []
        self.sword_dict = {}

        # 初始化玩家
        for i in range(self.knight_player_num):
            x = const.SCREEN_WIDTH // 3
            y = const.SCREEN_HEIGHT // 2
            self.knight_dict["knight_" + str(i)] = Knight(
                x, y, const.KNIGHT_SPEED, const.KNIGHT_HP
            )
            self.knight_list.append(self.knight_dict["knight_" + str(i)])

        for i in range(self.archer_player_num):
            x = const.SCREEN_WIDTH // 4
            y = const.SCREEN_HEIGHT // 2
            self.archer_dict["archer_" + str(i)] = Archer(
                x, y, const.ARCHER_SPEED, const.ARCHER_HP
            )
            self.archer_list.append(self.archer_dict["archer_" + str(i)])

        # 初始化奖励和信息
        self.rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

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

        reward = 0
        if agent in self.knight_dict:
            knight = self.knight_dict[agent]
            if knight.alive:
                reward = knight.update(action, self.knight_list, self.archer_list,
                                     self.zombie_list, self.arrow_list)
        elif agent in self.archer_dict:
            archer = self.archer_dict[agent]
            if archer.alive:
                reward = archer.update(action, self.knight_list, self.archer_list,
                                     self.zombie_list, self.arrow_list)

        # 生成僵尸
        if self.run_time % self.spawn_rate == 0 and len(self.zombie_list) < self.max_zombies:
            self._spawn_zombie()

        # 更新所有游戏对象
        self._update_game_objects()

        # 检查游戏是否结束
        alive_players = sum(knight.alive for knight in self.knight_list) + \
                       sum(archer.alive for archer in self.archer_list)
        if alive_players == 0:
            self.game_over = True

        # 更新奖励和状态
        self.rewards[agent] = reward
        if self.game_over:
            self.terminations = {agent: True for agent in self.agents}

        # 更新智能体选择
        self.agent_selection = self._agent_selector.next()

        # 渲染游戏
        self._render_game()

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
        if self.render_mode is None:
            return

        return np.copy(pygame.surfarray.array3d(self.render_surface).swapaxes(0, 1))

    def close(self):
        """关闭环境，释放资源。"""
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.quit()

    def _spawn_zombie(self):
        """生成一个新的僵尸。"""
        # 随机选择僵尸生成位置
        if var.rng.random() < 0.5:
            x = var.rng.integers(0, const.SCREEN_WIDTH)
            if var.rng.random() < 0.5:
                y = 0
            else:
                y = const.SCREEN_HEIGHT
        else:
            if var.rng.random() < 0.5:
                x = 0
            else:
                x = const.SCREEN_WIDTH
            y = var.rng.integers(0, const.SCREEN_HEIGHT)

        # 创建并添加僵尸
        zombie = Zombie(x, y)
        self.zombie_list.append(zombie)

    def _update_game_objects(self):
        """更新所有游戏对象的状态。"""
        # 更新箭矢
        for arrow in self.arrow_list:
            arrow.update()
            if not arrow.active:
                self.arrow_list.remove(arrow)

        # 更新僵尸
        for zombie in self.zombie_list:
            zombie.update(self.knight_list, self.archer_list)
            if not zombie.alive:
                self.zombie_list.remove(zombie)

        # 更新计时器
        self.run_time += 1

    def _render_game(self):
        """渲染游戏画面。"""
        # 清空屏幕
        self.render_surface.fill((0, 0, 0))

        # 绘制所有游戏对象
        for zombie in self.zombie_list:
            zombie.draw(self.render_surface)

        for arrow in self.arrow_list:
            if arrow.active:
                arrow.draw(self.render_surface)

        for knight in self.knight_list:
            if knight.alive:
                knight.draw(self.render_surface)

        for archer in self.archer_list:
            if archer.alive:
                archer.draw(self.render_surface)

        # 如果是人类模式，更新显示
        if self.render_mode == "human":
            self.window_surface.blit(self.render_surface, (0, 0))
            pygame.display.update()
