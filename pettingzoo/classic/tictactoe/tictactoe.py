# noqa: D212, D415
"""
井字棋游戏环境。

这个模块实现了标准的井字棋游戏，支持两个玩家轮流对弈。
游戏目标是在3x3的棋盘上先连成三个棋子。

# Tic Tac Toe

```{figure} classic_tictactoe.gif
:width: 140px
:name: tictactoe
```

这个环境是 <a href='..'>classic environments</a> 的一部分。请先阅读该页面的通用信息。

| Import             | `from pettingzoo.classic import tictactoe_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete                                      |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | `agents= ['player_1', 'player_2']`            |
| Agents             | 2                                             |
| Action Shape       | (1)                                           |
| Action Values      | [0, 8]                                        |
| Observation Shape  | (3, 3, 2)                                     |
| Observation Values | [0,1]                                         |

Tic-tac-toe 是一个简单的回合制策略游戏，两个玩家 X 和 O 轮流在 3x3 的棋盘上标记空间。第一个在水平、垂直或对角线上放置三个标记的玩家获胜。

### 观察空间

观察是一个字典，它包含一个 `'observation'` 元素，这是通常的 RL 观察，如下所述，以及一个 `'action_mask'`，它保存了合法的动作，如法律行动掩码部分所述。

主要观察是 3x3 棋盘的两个平面。对于玩家 1，第一个平面表示 X 的位置，第二个平面表示 O 的位置。每个单元格的可能值为 0 或 1；在第一个平面中，1 表示 X 已放置在该单元格中，0 表示 X 未放置在该单元格中。同样，在第二个平面中，1 表示 O 已放置在该单元格中，而 0 表示 O 未放置在该单元格中。对于玩家 2，观察相同，但 X 和 O 交换位置，因此 O 编码在第一个平面中，X 编码在第二个平面中。这允许自我对弈。

#### 合法动作掩码

当前智能体可用的合法动作在字典观察的 `action_mask` 元素中找到。 `action_mask` 是一个二进制向量，其中每个索引表示该动作是否合法。对于除了当前智能体之外的所有智能体，`action_mask` 都是零。采取非法动作会以 -1 的奖励结束游戏，对于非法移动的智能体，其他所有智能体的奖励为 0。

### 动作空间

每个动作从 0 到 8 都表示在相应单元格中放置 X 或 O。单元格索引如下：

```
0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8
```

### 奖励

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

如果游戏以平局结束，两个玩家都将获得 0 奖励。

### 版本历史

* v3: 修复了任意调用 observe() 的 bug (1.8.0)
* v2: 观察中的合法动作掩码取代了信息中的非法动作列表 (1.5.0)
* v1: 由于采用了新的智能体迭代方案，在所有智能体完成后迭代所有智能体，因此所有环境的版本都被提升了 (1.4.0)
* v0: 初始版本发布 (1.0.0)

"""
from __future__ import annotations

import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.classic.tictactoe.board import TTT_GAME_NOT_OVER, TTT_TIE, Board
from pettingzoo.utils import AgentSelector, wrappers


def get_image(path):
    """返回从给定路径加载的 pygame 图像。"""
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    return image


def get_font(path, size):
    """返回从给定路径加载的 pygame 字体。"""
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    font = pygame.font.Font((cwd + "/" + path), size)
    return font


def env(**kwargs):
    """创建井字棋环境实例。

    参数:
        render_mode (str, 可选): 渲染模式，可以是 "human" 或 None

    返回:
        TicTacToeEnv: 井字棋环境实例
    """
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    """井字棋环境的原始实现。

    这个类实现了完整的井字棋游戏逻辑，包括：
    1. 移动验证
    2. 状态更新
    3. 奖励计算
    4. 游戏结束判定

    属性:
        metadata (dict): 环境元数据
        possible_agents (list): 可能的智能体列表
        board (numpy.ndarray): 棋盘状态
        observation_spaces (dict): 每个智能体的观察空间
        action_spaces (dict): 每个智能体的动作空间
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "tictactoe_v3",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self, render_mode: str | None = None, screen_height: int | None = 1000
    ):
        """初始化井字棋环境。

        参数:
            render_mode (str, 可选): 渲染模式，可以是 "human" 或 None
            screen_height (int, 可选): 渲染模式为 "human" 时的屏幕高度
        """
        super().__init__()
        EzPickle.__init__(self, render_mode, screen_height)
        self.board = Board()

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(9) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(3, 3, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def observe(self, agent):
        """获取指定智能体的观察。

        参数:
            agent (str): 智能体名称

        返回:
            dict: 包含观察和动作掩码的字典
        """
        board_vals = np.array(self.board.squares).reshape(3, 3)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        observation = np.empty((3, 3, 2), dtype=np.int8)
        # 这将给出一个棋盘的副本，该棋盘对于玩家 1 的标记是 1，对于所有其他方格（无论是否为空）都是 0。
        observation[:, :, 0] = np.equal(board_vals, cur_player + 1)
        observation[:, :, 1] = np.equal(board_vals, opp_player + 1)

        action_mask = self._get_mask(agent)

        return {"observation": observation, "action_mask": action_mask}

    def _get_mask(self, agent):
        """获取指定智能体的合法动作掩码。

        参数:
            agent (str): 智能体名称

        返回:
            ndarray: 合法动作掩码
        """
        action_mask = np.zeros(9, dtype=np.int8)

        # 根据文档，除了当前选中的智能体之外的所有智能体的掩码都是全零。
        if agent == self.agent_selection:
            for i in self.board.legal_moves():
                action_mask[i] = 1

        return action_mask

    def observation_space(self, agent):
        """获取指定智能体的观察空间。

        参数:
            agent (str): 智能体名称

        返回:
            spaces.Dict: 观察空间
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """获取指定智能体的动作空间。

        参数:
            agent (str): 智能体名称

        返回:
            spaces.Discrete: 动作空间
        """
        return self.action_spaces[agent]

    # action 在这种情况下是一个值，从 0 到 8，表示在井字棋棋盘上的位置
    def step(self, action):
        """执行一步游戏。

        参数:
            action (int): 玩家的动作，表示在棋盘上的位置（0-8）

        返回:
            observations (dict): 每个玩家的观察
            rewards (dict): 每个玩家的奖励
            terminations (dict): 每个玩家的终止状态
            truncations (dict): 每个玩家的截断状态
            infos (dict): 每个玩家的额外信息
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.board.play_turn(self.agents.index(self.agent_selection), action)

        status = self.board.game_status()
        if status != TTT_GAME_NOT_OVER:
            if status == TTT_TIE:
                pass
            else:
                winner = status  # 要么是 TTT_PLAYER1_WIN，要么是 TTT_PLAYER2_WIN
                loser = winner ^ 1  # 0 -> 1; 1 -> 0
                self.rewards[self.agents[winner]] += 1
                self.rewards[self.agents[loser]] -= 1

            # 一旦任何玩家获胜或平局，游戏结束，两个玩家都完成了
            self.terminations = {i: True for i in self.agents}
            self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        """重置游戏到初始状态。

        参数:
            seed (int, 可选): 随机数种子
            options (dict, 可选): 重置选项

        返回:
            tuple: (observations, infos)
                - observations (dict): 每个智能体的初始观察
                - infos (dict): 每个智能体的初始信息
        """
        self.board.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # 选择第一个智能体
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if self.render_mode is not None and self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_height, self.screen_height)
            )
            pygame.display.set_caption("Tic-Tac-Toe")
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.screen_height, self.screen_height))

    def close(self):
        """关闭环境，释放资源。"""
        pass

    def render(self):
        """渲染当前游戏状态。

        根据 render_mode 的不同，可以：
        - "human": 在窗口中显示棋盘
        - "rgb_array": 返回 RGB 数组形式的棋盘图像

        返回:
            ndarray or None: 如果 render_mode 为 "rgb_array"，返回 RGB 数组；否则返回 None
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "您正在调用渲染方法，而没有指定任何渲染模式。"
            )
            return

        screen_height = self.screen_height
        screen_width = self.screen_height

        # 设置“X”和“O”标记的尺寸
        tile_size = int(screen_height / 4)

        # 加载并绘制棋盘图像
        board_img = get_image(os.path.join("img", "board.png"))
        board_img = pygame.transform.scale(
            board_img, (int(screen_width), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # 加载并绘制棋盘上的标记
        def getSymbol(input):
            if input == 0:
                return None
            elif input == 1:
                return "cross"
            else:
                return "circle"

        board_state = list(map(getSymbol, self.board.squares))

        mark_pos = 0
        for x in range(3):
            for y in range(3):
                mark = board_state[mark_pos]
                mark_pos += 1

                if mark is None:
                    continue

                mark_img = get_image(os.path.join("img", mark + ".png"))
                mark_img = pygame.transform.scale(mark_img, (tile_size, tile_size))

                self.screen.blit(
                    mark_img,
                    (
                        (screen_width / 3.1) * x + (screen_width / 17),
                        (screen_width / 3.145) * y + (screen_height / 19),
                    ),
                )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
