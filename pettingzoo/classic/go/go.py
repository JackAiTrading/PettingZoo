"""
围棋环境。

这个环境模拟了一个传统的围棋游戏，玩家轮流在棋盘上落子，
通过战略布局和战术对抗来围占更多的领地。

主要特点：
1. 双人对抗
2. 战略布局
3. 领地争夺
4. 战术对抗

环境规则：
1. 基本设置
   - 棋盘大小
   - 黑白棋子
   - 落子规则
   - 提子规则

2. 交互规则
   - 落子选择
   - 提子判定
   - 气的计算
   - 禁着处理

3. 智能体行为
   - 局势判断
   - 战略选择
   - 战术执行
   - 形势评估

4. 终止条件
   - 双方停子
   - 一方认输
   - 形势明朗
   - 终局计算

环境参数：
- 观察空间：棋盘状态
- 动作空间：落子位置选择
- 奖励：领地大小和胜负
- 最大步数：由规则设置决定

环境特色：
1. 棋盘系统
   - 位置关系
   - 气的计算
   - 提子规则
   - 死活判定

2. 规则机制
   - 落子规则
   - 提子规则
   - 禁着规则
   - 终局规则

3. 战术元素
   - 空间利用
   - 攻防转换
   - 地域争夺
   - 形势判断

4. 评估系统
   - 领地计算
   - 形势判断
   - 战术效果
   - 整体表现

注意事项：
- 局势判断
- 战略选择
- 战术执行
- 形势把握
"""

# noqa: D212, D415
r"""
# Go

```{figure} classic_go.gif
:width: 140px
:name: go
```

这个模块实现了标准的围棋游戏，支持两个玩家对弈。
游戏使用标准的围棋规则，包括提子、禁入点等规则。

这个环境是 PettingZoo 中的经典环境之一。请阅读 <a href='..'>classic environments</a> 页面以获取更多信息。

| Import             | `from pettingzoo.classic import go_v5` |
|--------------------|----------------------------------------|
| Actions            | Discrete                               |
| Parallel API       | Yes                                    |
| Manual Control     | No                                     |
| Agents             | `agents= ['black_0', 'white_0']`       |
| Agents             | 2                                      |
| Action Shape       | Discrete(362)                          |
| Action Values      | Discrete(362)                          |
| Observation Shape  | (19, 19, 3)                            |
| Observation Values | [0, 1]                                 |


Go 是一个棋盘游戏，两个玩家轮流下棋。黑棋先行，白棋后行。游戏的目标是控制棋盘上的更多区域，或者捕获对方的棋子。游戏结束时，如果双方连续两次跳过，则游戏结束。

我们的实现是对 [MiniGo](https://github.com/tensorflow/minigo) 的封装。

### 参数

Go 接受两个可选参数：棋盘大小（int）和贴目数（float）。默认值分别为 19 和 7.5。

```python
go_v5.env(board_size=19, komi=7.5)
```

`board_size`: 棋盘大小，通常为19。

`komi`: 贴目数，用于平衡黑白双方优势。7.5 是中国围棋比赛的标准值，但可能不是完全平衡的。

### 观察空间

观察空间是一个字典，包含 `'observation'` 和 `'action_mask'` 两个元素。

*   `'observation'`: 棋盘状态，形状为 `(N, N, 3)`，其中 `N` 是棋盘大小。第一个平面表示当前玩家的棋子，第二个平面表示对方的棋子，第三个平面表示当前玩家是黑棋还是白棋。
*   `'action_mask'`: 合法动作掩码，形状为 `(N^2 + 1,)`，其中 `N` 是棋盘大小。每个元素表示对应位置是否是合法动作。

### 动作空间

动作空间是离散的，形状为 `(N^2 + 1,)`，其中 `N` 是棋盘大小。每个元素表示对应位置的动作。

### 奖励

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

### 版本历史

*   v5: 更改观察空间为 AlphaZero 风格的帧堆叠 (1.11.0)
*   v4: 修复黑白棋子在观察空间中的保存问题 (1.10.0)
*   v3: 修复任意调用 observe() 的问题 (1.8.0)
*   v2: 在观察空间中使用合法动作掩码替代非法动作列表 (1.5.0)
*   v1: 更新所有环境以采用新的智能体迭代方案 (1.4.0)
*   v0: 初始版本 (1.0.0)

"""
from __future__ import annotations

import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.classic.go import coords, go_base
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector


def get_image(path):
    from os import path as os_path

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
    """围棋游戏的主要环境类。

    这个环境实现了标准的围棋游戏，支持两个玩家轮流对弈。

    属性:
        metadata (dict): 环境的元数据，包括版本信息和渲染模式
        possible_agents (list): 可能的智能体列表，包括黑方和白方
        board_size (int): 棋盘大小，通常为19
        komi (float): 贴目数，用于平衡黑白双方优势
        board (ndarray): 棋盘状态
        action_spaces (dict): 每个玩家的动作空间
        observation_spaces (dict): 每个玩家的观察空间
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "go_v5",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(
        self,
        board_size: int = 19,
        komi: float = 7.5,
        render_mode: str | None = None,
        screen_height: int | None = 800,
    ):
        """初始化围棋环境。

        参数:
            board_size (int, 可选): 棋盘大小，默认为19
            komi (float, 可选): 贴目数，默认为7.5
            render_mode (str, 可选): 渲染模式，可以是 "human" 或 "rgb_array"
        """
        EzPickle.__init__(self, board_size, komi, render_mode, screen_height)
        # board_size: a int, representing the board size (board has a board_size x board_size shape)
        # komi: a float, representing points given to the second player.
        super().__init__()

        self._overwrite_go_global_variables(board_size=board_size)
        self._komi = komi

        self.agents = ["black_0", "white_0"]
        self.possible_agents = self.agents[:]

        self.screen = None

        self.observation_spaces = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        "observation": spaces.Box(
                            low=0, high=1, shape=(self._N, self._N, 17), dtype=bool
                        ),
                        "action_mask": spaces.Box(
                            low=0,
                            high=1,
                            shape=((self._N * self._N) + 1,),
                            dtype=np.int8,
                        ),
                    }
                )
                for _ in range(self.num_agents)
            ]
        )

        self.action_spaces = self._convert_to_dict(
            [spaces.Discrete(self._N * self._N + 1) for _ in range(self.num_agents)]
        )

        self._agent_selector = AgentSelector(self.agents)

        self.board_history = np.zeros((self._N, self._N, 16), dtype=bool)

        self.render_mode = render_mode
        self.screen_width = self.screen_height = screen_height

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def observation_space(self, agent):
        """获取指定玩家的观察空间。

        参数:
            agent (str): 玩家标识符

        返回:
            spaces.Dict: 玩家的观察空间
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """获取指定玩家的动作空间。

        参数:
            agent (str): 玩家标识符

        返回:
            spaces.Discrete: 玩家的动作空间
        """
        return self.action_spaces[agent]

    def _overwrite_go_global_variables(self, board_size: int):
        """覆盖 Go 全局变量。

        参数:
            board_size (int): 棋盘大小
        """
        self._N = board_size
        go_base.N = self._N
        go_base.ALL_COORDS = [(i, j) for i in range(self._N) for j in range(self._N)]
        go_base.EMPTY_BOARD = np.zeros([self._N, self._N], dtype=np.int8)
        go_base.NEIGHBORS = {
            (x, y): list(
                filter(
                    self._check_bounds, [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                )
            )
            for x, y in go_base.ALL_COORDS
        }
        go_base.DIAGONALS = {
            (x, y): list(
                filter(
                    self._check_bounds,
                    [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)],
                )
            )
            for x, y in go_base.ALL_COORDS
        }
        return

    def _check_bounds(self, c):
        """检查位置是否在棋盘范围内。

        参数:
            c (tuple): 位置

        返回:
            bool: 是否在棋盘范围内
        """
        return 0 <= c[0] < self._N and 0 <= c[1] < self._N

    def _encode_player_plane(self, agent):
        """编码玩家平面。

        参数:
            agent (str): 玩家标识符

        返回:
            ndarray: 玩家平面
        """
        if agent == self.possible_agents[0]:
            return np.zeros([self._N, self._N], dtype=bool)
        else:
            return np.ones([self._N, self._N], dtype=bool)

    def _encode_board_planes(self, agent):
        """编码棋盘平面。

        参数:
            agent (str): 玩家标识符

        返回:
            ndarray: 棋盘平面
        """
        agent_factor = (
            go_base.BLACK if agent == self.possible_agents[0] else go_base.WHITE
        )
        current_agent_plane_idx = np.where(self._go.board == agent_factor)
        opponent_agent_plane_idx = np.where(self._go.board == -agent_factor)
        current_agent_plane = np.zeros([self._N, self._N], dtype=bool)
        opponent_agent_plane = np.zeros([self._N, self._N], dtype=bool)
        current_agent_plane[current_agent_plane_idx] = 1
        opponent_agent_plane[opponent_agent_plane_idx] = 1
        return current_agent_plane, opponent_agent_plane

    def _int_to_name(self, ind):
        """将索引转换为玩家名称。

        参数:
            ind (int): 索引

        返回:
            str: 玩家名称
        """
        return self.possible_agents[ind]

    def _name_to_int(self, name):
        """将玩家名称转换为索引。

        参数:
            name (str): 玩家名称

        返回:
            int: 索引
        """
        return self.possible_agents.index(name)

    def _convert_to_dict(self, list_of_list):
        """将列表转换为字典。

        参数:
            list_of_list (list): 列表

        返回:
            dict: 字典
        """
        return dict(zip(self.possible_agents, list_of_list))

    def _encode_legal_actions(self, actions):
        """编码合法动作。

        参数:
            actions (ndarray): 动作

        返回:
            ndarray: 合法动作
        """
        return np.where(actions == 1)[0]

    def _encode_rewards(self, result):
        """编码奖励。

        参数:
            result (int): 结果

        返回:
            list: 奖励
        """
        return [1, -1] if result == 1 else [-1, 1]

    def observe(self, agent):
        """获取指定玩家的观察。

        参数:
            agent (str): 玩家标识符

        返回:
            dict: 观察
        """
        current_agent_plane, opponent_agent_plane = self._encode_board_planes(agent)
        player_plane = self._encode_player_plane(agent)

        observation = np.dstack((self.board_history, player_plane))

        legal_moves = self.next_legal_moves if agent == self.agent_selection else []
        action_mask = np.zeros((self._N * self._N) + 1, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def step(self, action):
        """执行一步游戏。

        参数:
            action (int): 动作

        返回:
            observations (dict): 观察
            rewards (dict): 奖励
            terminations (dict): 终止状态
            truncations (dict): 截断状态
            infos (dict): 额外信息
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        self._go = self._go.play_move(coords.from_flat(action))
        self._last_obs = self.observe(self.agent_selection)
        current_agent_plane, opponent_agent_plane = self._encode_board_planes(
            self.agent_selection
        )
        self.board_history = np.dstack(
            (current_agent_plane, opponent_agent_plane, self.board_history[:, :, :-2])
        )
        next_player = self._agent_selector.next()
        if self._go.is_game_over():
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.rewards = self._convert_to_dict(
                self._encode_rewards(self._go.result())
            )
            self.next_legal_moves = [self._N * self._N]
        else:
            self.next_legal_moves = self._encode_legal_actions(
                self._go.all_legal_moves()
            )
        self.agent_selection = (
            next_player if next_player else self._agent_selector.next()
        )
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        """重置游戏到初始状态。

        参数:
            seed (int, 可选): 随机种子
            options (dict, 可选): 重置选项

        返回:
            observations (dict): 观察
            infos (dict): 额外信息
        """
        self._go = go_base.Position(board=None, komi=self._komi)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.terminations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.truncations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.infos = self._convert_to_dict([{} for _ in range(self.num_agents)])
        self.next_legal_moves = self._encode_legal_actions(self._go.all_legal_moves())
        self._last_obs = self.observe(self.agents[0])
        self.board_history = np.zeros((self._N, self._N, 16), dtype=bool)

    def render(self):
        """渲染当前游戏状态。

        根据render_mode的不同，可以：
        - "human": 在窗口中显示棋盘
        - "rgb_array": 返回RGB数组形式的棋盘图像

        返回:
            ndarray or None: 如果render_mode为"rgb_array"，返回RGB数组；否则返回None
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Go")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        size = go_base.N

        # Load and scale all of the necessary images
        tile_size = self.screen_width / size

        black_stone = get_image(os.path.join("img", "GoBlackPiece.png"))
        black_stone = pygame.transform.scale(
            black_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6)))
        )

        white_stone = get_image(os.path.join("img", "GoWhitePiece.png"))
        white_stone = pygame.transform.scale(
            white_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6)))
        )

        tile_img = get_image(os.path.join("img", "GO_Tile0.png"))
        tile_img = pygame.transform.scale(
            tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6)))
        )

        # blit board tiles
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                self.screen.blit(tile_img, ((i * (tile_size)), int(j) * (tile_size)))

        for i in range(1, 9):
            tile_img = get_image(os.path.join("img", "GO_Tile" + str(i) + ".png"))
            tile_img = pygame.transform.scale(
                tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6)))
            )
            for j in range(1, size - 1):
                if i == 1:
                    self.screen.blit(tile_img, (0, int(j) * (tile_size)))
                elif i == 2:
                    self.screen.blit(tile_img, ((int(j) * (tile_size)), 0))
                elif i == 3:
                    self.screen.blit(
                        tile_img, ((size - 1) * (tile_size), int(j) * (tile_size))
                    )
                elif i == 4:
                    self.screen.blit(
                        tile_img, ((int(j) * (tile_size)), (size - 1) * (tile_size))
                    )
            if i == 5:
                self.screen.blit(tile_img, (0, 0))
            elif i == 6:
                self.screen.blit(tile_img, ((size - 1) * (tile_size), 0))
            elif i == 7:
                self.screen.blit(
                    tile_img, ((size - 1) * (tile_size), (size - 1) * (tile_size))
                )
            elif i == 8:
                self.screen.blit(tile_img, (0, (size - 1) * (tile_size)))

        offset = tile_size * (1 / 6)
        # Blit the necessary chips and their positions
        for i in range(0, size):
            for j in range(0, size):
                if self._go.board[i][j] == go_base.BLACK:
                    self.screen.blit(
                        black_stone,
                        ((i * (tile_size) + offset), int(j) * (tile_size) + offset),
                    )
                elif self._go.board[i][j] == go_base.WHITE:
                    self.screen.blit(
                        white_stone,
                        ((i * (tile_size) + offset), int(j) * (tile_size) + offset),
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

    def close(self):
        """关闭环境，释放资源。"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _check_liberty(self, position):
        """检查指定位置的棋子是否有气。

        参数:
            position (tuple): 位置

        返回:
            bool: 是否有气
        """
        return go_base.liberty(position)

    def _remove_group(self, position):
        """移除指定位置的棋子组。

        参数:
            position (tuple): 位置

        返回:
            set: 被移除的所有棋子的位置
        """
        return go_base.remove_group(position)
