"""
四子棋环境。

这个环境模拟了一个经典的四子棋游戏，玩家轮流在垂直的棋盘上
放置棋子，通过战略布局来首先连成四个棋子获得胜利。

主要特点：
1. 双人对抗
2. 战略布局
3. 空间控制
4. 战术规划

环境规则：
1. 基本设置
   - 棋盘大小
   - 棋子颜色
   - 落子规则
   - 胜负判定

2. 交互规则
   - 选择列数
   - 棋子下落
   - 连线判定
   - 回合交替

3. 智能体行为
   - 位置选择
   - 战术布局
   - 防守反击
   - 胜势判断

4. 终止条件
   - 连成四子
   - 棋盘已满
   - 认输投降
   - 和局判定

环境参数：
- 观察空间：棋盘状态
- 动作空间：列数选择
- 奖励：获胜和战术奖励
- 最大步数：由棋盘大小决定

环境特色：
1. 棋盘系统
   - 垂直结构
   - 重力效应
   - 连线检测
   - 状态记录

2. 控制机制
   - 列数选择
   - 落子确认
   - 有效性检查
   - 回合管理

3. 战术元素
   - 空间利用
   - 连线威胁
   - 防守布局
   - 进攻策略

4. 评估系统
   - 局势评估
   - 威胁程度
   - 战术效果
   - 整体表现

注意事项：
- 位置选择
- 战术布局
- 防守意识
- 进攻时机
"""
# noqa: D212, D415
"""
# 四子棋游戏环境

这个模块实现了标准的四子棋游戏，支持两个玩家对弈。
游戏目标是在垂直、水平或对角线方向上先连成四个棋子。

```{figure} classic_connect_four.gif
:width: 140px
:name: connect_four
```

这个环境是 <a href='..'>经典环境</a> 的一部分。请先阅读该页面以获取一般信息。

| 导入             | `from pettingzoo.classic import connect_four_v3` |
|--------------------|--------------------------------------------------|
| 动作            | 离散                                         |
| 并行 API       | 是                                              |
| 手动控制     | 否                                               |
| 智能体             | `agents= ['player_0', 'player_1']`               |
| 智能体             | 2                                                |
| 动作形状       | (1,)                                             |
| 动作值        | 离散(7)                                      |
| 观察形状       | (6, 7, 2)                                        |
| 观察值        | [0,1]                                            |


四子棋是两个玩家轮流下棋的游戏，玩家必须在垂直、水平或对角线方向上连成四个棋子。玩家将他们的棋子掉落在棋盘的某一列中，棋子会掉落到棋盘底部或停留在现有的棋子上。玩家不能在满列中放置棋子，游戏结束时要么有玩家连成四个棋子，要么所有七列都被填满。

### 观察空间

观察是一个字典，它包含一个 `'observation'` 元素，这是通常的 RL 观察，如下所述，还有一个 `'action_mask'`，它保存了合法动作，如“合法动作掩码”部分所述。

主观察空间是 6x7 网格的两个平面。每个平面代表特定玩家的棋子，每个网格位置代表相应玩家的棋子放置位置。1 表示玩家在该单元格中放置了棋子，0 表示他们没有在该单元格中放置棋子。0 表示单元格为空，或者另一个玩家在该单元格中放置了棋子。

#### 合法动作掩码

当前玩家可用的合法动作在字典观察的 `action_mask` 元素中找到。`action_mask` 是一个二进制向量，其中每个索引代表动作是否合法。对于除了当前玩家之外的所有玩家，`action_mask` 都是全零。执行非法动作会以 -1 的奖励结束游戏，对于非法移动的玩家，其他玩家会获得 0 的奖励。

### 动作空间

动作空间是从 0 到 6（含）的整数集，其中动作表示应该在哪一列下落棋子。

### 奖励

如果玩家成功连接四个棋子，他们将获得 1 分奖励。同时，对方玩家将获得 -1 分奖励。如果游戏结束时平局，双方玩家都将获得 0 分奖励。

### 版本历史

* v3: 修复了任意调用 observe() 的 bug (1.8.0)
* v2: 观察中的合法动作掩码替换了信息中的非法动作列表 (1.5.0)
* v1: 由于采用了新的智能体迭代方案，所有环境的版本都被提升了，在该方案中，所有智能体在完成后都会被迭代 (1.4.0)
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
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector


def get_image(path):
    from os import path as os_path

    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    """四子棋游戏的主要环境类。

    这个环境实现了标准的四子棋游戏，支持两个玩家轮流对弈。

    属性：
        metadata (dict): 环境的元数据，包括版本信息和渲染模式
        possible_agents (list): 可能的智能体列表，包括玩家1和玩家2
        board (ndarray): 棋盘状态，6x7的二维数组
        action_spaces (dict): 每个玩家的动作空间
        observation_spaces (dict): 每个玩家的观察空间
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "connect_four_v3",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str | None = None, screen_scaling: int = 9):
        """初始化四子棋环境。

        参数：
            render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"
            screen_scaling (int, 可选): 屏幕缩放因子，默认为 9
        """
        EzPickle.__init__(self, render_mode, screen_scaling)
        super().__init__()
        # 6 行 x 7 列
        # 空白格子 = 0
        # 玩家 0 -- 1
        # 玩家 1 -- 2
        # 平面表示法中的行主序
        self.screen = None
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling

        self.board = [0] * (6 * 7)

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(7) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(6, 7, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    # 键
    # ----
    # 空白格子 = 0
    # 玩家 0 = 1
    # 玩家 1 = 2
    # 观察是一个列表，其中每个列表代表一行
    #
    # 数组([[0, 1, 1, 2, 0, 1, 0],
    #        [1, 0, 1, 2, 2, 2, 1],
    #        [0, 1, 0, 0, 1, 2, 1],
    #        [1, 0, 2, 0, 1, 1, 0],
    #        [2, 0, 0, 0, 1, 1, 0],
    #        [1, 1, 2, 1, 0, 1, 0]], dtype=int8)
    def observe(self, agent):
        """获取指定玩家的观察。

        参数：
            agent (str): 玩家标识符

        返回：
            dict: 玩家的观察，包括：
                - observation: 棋盘状态
                - action_mask: 合法动作掩码
        """
        board_vals = np.array(self.board).reshape(6, 7)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)

        observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        legal_moves = self._legal_moves() if agent == self.agent_selection else []

        action_mask = np.zeros(7, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def observation_space(self, agent):
        """获取指定玩家的观察空间。

        参数：
            agent (str): 玩家标识符

        返回：
            spaces.Dict: 玩家的观察空间
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """获取指定玩家的动作空间。

        参数：
            agent (str): 玩家标识符

        返回：
            spaces.Discrete: 玩家的动作空间
        """
        return self.action_spaces[agent]

    def _legal_moves(self):
        """获取当前玩家的合法动作。

        返回：
            list: 合法动作列表
        """
        return [i for i in range(7) if self.board[i] == 0]

    # 动作在本例中是一个值，从 0 到 6，表示在平面表示法中的棋盘上移动的位置
    def step(self, action):
        """执行一步游戏。

        参数：
            action (int): 玩家的动作，表示在哪一列下落棋子（0-6）

        返回：
            observations (dict): 每个玩家的观察
            rewards (dict): 每个玩家的奖励
            terminations (dict): 每个玩家的终止状态
            truncations (dict): 每个玩家的截断状态
            infos (dict): 每个玩家的额外信息
        """
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        # 断言有效动作
        assert self.board[0:7][action] == 0, "玩家执行了非法动作。"

        piece = self.agents.index(self.agent_selection) + 1
        for i in list(filter(lambda x: x % 7 == action, list(range(41, -1, -1)))):
            if self.board[i] == 0:
                self.board[i] = piece
                break

        next_agent = self._agent_selector.next()

        winner = self.check_for_winner()

        # 检查是否有玩家获胜
        if winner:
            self.rewards[self.agent_selection] += 1
            self.rewards[next_agent] -= 1
            self.terminations = {i: True for i in self.agents}
        # 检查是否平局
        elif all(x in [1, 2] for x in self.board):
            # 一旦任何玩家获胜或平局，游戏结束，所有玩家都完成了
            self.terminations = {i: True for i in self.agents}

        self.agent_selection = next_agent

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        """重置游戏到初始状态。

        参数：
            seed (int, 可选): 随机种子
            options (dict, 可选): 重置选项

        返回：
            observations (dict): 每个玩家的初始观察
            infos (dict): 每个玩家的初始信息
        """
        # 重置环境
        self.board = [0] * (6 * 7)

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = AgentSelector(self.agents)

        self.agent_selection = self._agent_selector.reset()

    def render(self):
        """渲染当前游戏状态。

        根据 render_mode 的不同，可以：
        - "human": 在窗口中显示棋盘
        - "rgb_array": 返回 RGB 数组形式的棋盘图像

        返回：
            ndarray or None: 如果 render_mode 为 "rgb_array"，返回 RGB 数组；否则返回 None
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "您正在调用渲染方法，而没有指定任何渲染模式。"
            )
            return

        screen_width = 99 * self.screen_scaling
        screen_height = 86 / 99 * screen_width

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("四子棋")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((screen_width, screen_height))

        # 加载和缩放所有必要的图像
        tile_size = (screen_width * (91 / 99)) / 7

        red_chip = get_image(os.path.join("img", "C4RedPiece.png"))
        red_chip = pygame.transform.scale(
            red_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        black_chip = get_image(os.path.join("img", "C4BlackPiece.png"))
        black_chip = pygame.transform.scale(
            black_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        board_img = get_image(os.path.join("img", "Connect4Board.png"))
        board_img = pygame.transform.scale(
            board_img, ((int(screen_width)), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # 绘制必要的棋子及其位置
        for i in range(0, 42):
            if self.board[i] == 1:
                self.screen.blit(
                    red_chip,
                    (
                        (i % 7) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 7) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )
            elif self.board[i] == 2:
                self.screen.blit(
                    black_chip,
                    (
                        (i % 7) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 7) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        """关闭环境，释放资源。"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def check_for_winner(self):
        """检查是否有玩家获胜。

        检查水平、垂直和对角线方向是否有四个相连的棋子。

        返回：
            bool: 是否有玩家获胜
        """
        board = np.array(self.board).reshape(6, 7)
        piece = self.agents.index(self.agent_selection) + 1

        # 检查水平位置是否获胜
        column_count = 7
        row_count = 6

        for c in range(column_count - 3):
            for r in range(row_count):
                if (
                    board[r][c] == piece
                    and board[r][c + 1] == piece
                    and board[r][c + 2] == piece
                    and board[r][c + 3] == piece
                ):
                    return True

        # 检查垂直位置是否获胜
        for c in range(column_count):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c] == piece
                    and board[r + 2][c] == piece
                    and board[r + 3][c] == piece
                ):
                    return True

        # 检查正斜率对角线是否获胜
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c + 1] == piece
                    and board[r + 2][c + 2] == piece
                    and board[r + 3][c + 3] == piece
                ):
                    return True

        # 检查负斜率对角线是否获胜
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if (
                    board[r][c] == piece
                    and board[r - 1][c + 1] == piece
                    and board[r - 2][c + 2] == piece
                    and board[r - 3][c + 3] == piece
                ):
                    return True

        return False
