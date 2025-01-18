"""
国际象棋环境模块。

这个模块实现了双人国际象棋游戏环境。游戏使用标准的国际象棋规则，
包括所有特殊规则（王车易位、吃过路兵、升变等）。

环境特点：
1. 支持标准的国际象棋规则
2. 提供完整的游戏状态观察
3. 实现了合法移动验证
4. 支持游戏结果判定（将军、和棋等）
"""

import os

import chess
import numpy as np
import pygame

from pettingzoo.classic.chess import chess_utils
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv


def env(render_mode=None):
    """创建国际象棋环境实例。

    参数:
        render_mode (str, 可选): 渲染模式，可以是 "human" 或 None

    返回:
        ChessEnv: 国际象棋环境实例
    """
    env = raw_env(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """国际象棋环境的原始实现。

    这个类实现了完整的国际象棋游戏逻辑，包括：
    1. 移动验证
    2. 状态更新
    3. 奖励计算
    4. 游戏结束判定

    属性:
        metadata (dict): 环境元数据
        possible_agents (list): 可能的智能体列表
        board (chess.Board): 棋盘状态
        screen (pygame.Surface): 游戏画面
        clock (pygame.time.Clock): 游戏时钟
    """

    metadata = {
        "render_modes": ["human"],
        "name": "chess_v5",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode=None):
        """初始化国际象棋环境。

        参数:
            render_mode (str, 可选): 渲染模式
        """
        super().__init__()

        self.possible_agents = ["player_0", "player_1"]

        self.action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.possible_agents}
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(8, 8, 111), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(8 * 8 * 73,), dtype=np.int8
                    ),
                }
            )
            for name in self.possible_agents
        }

        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.window_size = 512

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

    def render(self):
        """渲染当前游戏状态。

        如果渲染模式为 "human"，将在窗口中显示棋盘。

        返回:
            pygame.Surface: 渲染的游戏画面
        """
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            if os.path.exists(os.path.join(os.path.dirname(__file__), "assets")):
                self.chess_img = pygame.image.load(
                    os.path.join(os.path.dirname(__file__), "assets", "chess_board.png")
                )
                self.chess_img = pygame.transform.scale(
                    self.chess_img, (self.window_size, self.window_size)
                )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        if os.path.exists(os.path.join(os.path.dirname(__file__), "assets")):
            canvas.blit(self.chess_img, (0, 0))
            # 在这里添加绘制棋子的代码

        self.screen.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        return canvas

    def close(self):
        """关闭环境，释放资源。"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def reset(self, seed=None, options=None):
        """重置环境到初始状态。

        参数:
            seed (int, 可选): 随机数种子
            options (dict, 可选): 重置选项

        返回:
            tuple: (observations, infos)
                - observations (dict): 每个智能体的初始观察
                - infos (dict): 每个智能体的初始信息
        """
        self.agents = self.possible_agents[:]
        self.board = chess.Board()
        self.num_moves = 0
        self.agent_selection = self.agents[0]

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        return self._get_obs(), self.infos

    def step(self, action):
        """执行一步环境交互。

        参数:
            action (int): 选择的动作

        返回:
            None: 环境状态会被更新，但不直接返回
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        next_index = (current_index + 1) % len(self.agents)
        self.agent_selection = self.agents[next_index]

        move = chess_utils.action_to_move(self.board, action)
        self.board.push(move)

        next_legal_moves = list(self.board.legal_moves)

        is_stale_or_checkmate = (
            self.board.is_stalemate()
            or self.board.is_checkmate()
            or self.board.is_insufficient_material()
            or self.board.can_claim_draw()
        )

        # 处理游戏结束情况
        if is_stale_or_checkmate:
            self.rewards[self.agents[0]] = 0
            self.rewards[self.agents[1]] = 0

            # 处理将军结束
            if self.board.is_checkmate():
                # 确定获胜者
                winner = current_index
                self.rewards[self.agents[winner]] = 1
                self.rewards[self.agents[1 - winner]] = -1

            for agent in self.agents:
                self.terminations[agent] = True

        self._accumulate_rewards()

        self.num_moves += 1
