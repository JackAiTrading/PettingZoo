"""
国际象棋游戏环境。

这个模块实现了标准的国际象棋游戏，支持两个玩家对弈。
游戏规则遵循国际象棋联合会(FIDE)的标准规则。

这个环境是经典环境的一部分。请先阅读该页面的通用信息。

| 导入             | `from pettingzoo.classic import chess_v6` |
|--------------------|------------------------------------------|
| 动作类型          | 离散                                     |
| 并行API          | 是                                       |
| 手动控制         | 否                                       |
| 智能体           | `agents= ['player_0', 'player_1']`       |
| 智能体数量       | 2                                        |
| 动作形状         | Discrete(4672)                           |
| 动作值           | Discrete(4672)                           |
| 观察形状         | (8, 8, 111)                             |
| 观察值           | [0, 1]                                   |


## 描述

国际象棋是一个双人回合制策略游戏。游戏在8x8的棋盘上进行，每个玩家开始时有16个棋子：1个王、1个后、2个车、2个象、2个马和8个兵。玩家轮流移动棋子，目标是将死对方的王。

### 观察空间

观察是一个字典，包含一个 `'observation'` 元素（即下面描述的常规RL观察）和一个 `'action_mask'`（包含合法动作，在合法动作掩码部分描述）。

主要观察空间是一个8x8x111的三维数组。前108个平面表示棋盘的当前状态和历史状态，最后3个平面表示重复动作计数、半回合数和全回合数。

每个位置状态由以下特征表示：
- 棋子类型（王、后、车、象、马、兵）
- 棋子颜色（白、黑）
- 重复动作计数
- 半回合数（自上次吃子或推兵以来的回合数）
- 全回合数

#### 合法动作掩码

当前智能体可用的合法动作可以在字典观察的 `action_mask` 元素中找到。`action_mask` 是一个二进制向量，其中每个索引表示该动作是否合法。除了当前轮到的智能体外，其他所有智能体的 `action_mask` 都是零。采取非法动作会导致游戏结束，非法行动的智能体获得-1的奖励，其他所有智能体获得0的奖励。

### 动作空间

动作空间是一个离散空间，大小为4672，表示所有可能的棋子移动。每个动作编码了：
- 起始位置（从0到63）
- 目标位置（从0到63）
- 升变类型（如果是兵升变）

### 奖励

|   结果   | 赢家 | 输家 |
|:--------:|:----:|:----:|
| 将死     | +1   | -1   |
| 和棋     |  0   |  0   |
| 非法移动 | -1   |  0   |

### 版本历史

* v6: 修复了任意调用 observe() 的错误，添加了chess-draw环境 (1.8.0)
* v5: 添加了合法动作掩码 (1.5.0)
* v4: 修复了观察空间 (1.4.0)
* v3: 修复了重置函数 (1.3.1)
* v2: 修复了渲染 (1.2.0)
* v1: 修复了观察空间和文档 (1.1.0)
* v0: 初始版本发布 (1.0.0)
"""

import os

import chess
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv


def get_image(path):
    """获取指定路径的图像。

    参数:
        path (str): 图像文件路径

    返回:
        pygame.Surface: 加载的图像
    """
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    return image


def env(render_mode=None):
    """创建国际象棋环境的包装器。

    参数:
        render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"

    返回:
        AECEnv: 包装后的环境
    """
    env = raw_env(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    """国际象棋游戏的主要环境类。

    这个环境实现了标准的国际象棋游戏，支持两个玩家对弈。

    属性:
        metadata (dict): 环境的元数据，包括版本信息和渲染模式
        possible_agents (list): 可能的智能体列表，包括玩家1和玩家2
        board (chess.Board): 棋盘状态
        action_spaces (dict): 每个玩家的动作空间
        observation_spaces (dict): 每个玩家的观察空间
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "chess_v6",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode=None):
        """初始化国际象棋环境。

        参数:
            render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"
        """
        super().__init__()
        EzPickle.__init__(self, render_mode)

        self.board = chess.Board()

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.agents}
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(8, 8, 111), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(4672,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }

        self.render_mode = render_mode

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
            pygame.init()
            pygame.display.init()
            self.window_surface = None
            self.chess_surface = None

    def observation_space(self, agent):
        """获取指定智能体的观察空间。

        参数:
            agent (str): 智能体标识符

        返回:
            spaces.Dict: 观察空间
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

    def reset(self, seed=None, options=None):
        """重置游戏到初始状态。

        参数:
            seed (int, 可选): 随机种子
            options (dict, 可选): 重置选项

        返回:
            observations (dict): 每个玩家的初始观察
            infos (dict): 每个玩家的初始信息
        """
        self.agents = self.possible_agents[:]

        self.board = chess.Board()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        observations = {name: self.observe(name) for name in self.agents}

        return observations, self.infos

    def step(self, action):
        """执行一步动作。

        参数:
            action (int): 要执行的动作

        返回:
            observations (dict): 每个玩家的新观察
            rewards (dict): 每个玩家的奖励
            terminations (dict): 每个玩家的终止状态
            truncations (dict): 每个玩家的截断状态
            infos (dict): 每个玩家的信息
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)

        # Cast action into int
        action = int(action)

        chosen_move = chess_utils.action_to_move(self.board, action, current_index)
        assert chosen_move in self.board.legal_moves
        self.board.push(chosen_move)

        next_legal_moves = chess_utils.legal_moves(self.board)

        is_stale_or_checkmate = not any(next_legal_moves)

        # claim draw is set to be true to align with normal tournament rules
        is_insufficient_material = self.board.is_insufficient_material()
        can_claim_draw = self.board.can_claim_draw()
        game_over = can_claim_draw or is_stale_or_checkmate or is_insufficient_material

        if game_over:
            result = self.board.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            self.set_game_result(result_val)

        self._accumulate_rewards()

        # Update board after applying action
        # We always take the perspective of the white agent
        next_board = chess_utils.get_observation(self.board, player=0)
        self.board_history = np.dstack(
            (next_board[:, :, 7:], self.board_history[:, :, :-13])
        )
        self.agent_selection = (
            self._agent_selector.next()
        )  # Give turn to the next agent

        observations = {name: self.observe(name) for name in self.agents}

        return (
            observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def observe(self, agent):
        """获取指定智能体的观察。

        参数:
            agent (str): 智能体标识符

        返回:
            dict: 观察
        """
        current_index = self.possible_agents.index(agent)

        observation = chess_utils.get_observation(self.board, current_index)
        observation = np.dstack((observation[:, :, :7], self.board_history))
        # We need to swap the white 6 channels with black 6 channels
        if current_index == 1:
            # 1. Mirror the board
            observation = np.flip(observation, axis=0)
            # 2. Swap the white 6 channels with the black 6 channels
            for i in range(1, 9):
                tmp = observation[..., 13 * i - 6 : 13 * i].copy()
                observation[..., 13 * i - 6 : 13 * i] = observation[
                    ..., 13 * i : 13 * i + 6
                ]
                observation[..., 13 * i : 13 * i + 6] = tmp
        legal_moves = (
            chess_utils.legal_moves(self.board) if agent == self.agent_selection else []
        )

        action_mask = np.zeros(4672, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def set_game_result(self, result_val):
        """设置游戏结果。

        参数:
            result_val (int): 结果值（1表示白方胜，-1表示黑方胜，0表示平局）
        """
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {"legal_moves": []}

    def render(self):
        """渲染当前游戏状态。

        根据渲染模式返回不同的表示：
        - "ansi"：返回棋盘的字符串表示
        - "human"：在窗口中显示棋盘
        - "rgb_array"：返回RGB数组表示的棋盘图像
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def _render_gui(self):
        """渲染图形用户界面。

        这是一个内部方法，用于在 pygame 窗口中渲染棋盘。
        """
        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("Chess")
                self.window_surface = pygame.display.set_mode((800, 800))
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface((800, 800))

        self.window_surface.fill((255, 255, 255))

        for square, piece in self.board.piece_map().items():
            pos_x = square % 8 * 100
            pos_y = 800 - (square // 8 + 1) * 100
            piece_name = chess.piece_name(piece.piece_type)
            piece_img = get_image(f"{piece_name}_{piece.color}.png")
            self.window_surface.blit(piece_img, (pos_x, pos_y))

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def close(self):
        """关闭环境，释放资源。

        关闭 pygame 窗口并清理相关资源。
        """
        if self.window_surface is not None:
            pygame.quit()
            self.window_surface = None
