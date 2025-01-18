import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

ROCK = 0
PAPER = 1
SCISSORS = 2
NO_MOVE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


def env(render_mode=None):
    """
    env 函数通常默认用包装器包装环境。
    你可以在开发者文档的其他部分找到这些方法的完整文档。
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # 这个包装器仅用于向终端打印结果的环境
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # 这个包装器帮助处理离散动作空间的错误
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # 提供各种有用的用户错误提示
    # 强烈推荐使用
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    为了支持 AEC API，raw_env() 函数只是使用 from_parallel
    函数将 ParallelEnv 转换为 AEC 环境
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        """
        init 方法接收环境参数，并且应该定义以下属性：
        - possible_agents
        - render_mode

        注意：从 v1.18.1 开始，action_spaces 和 observation_spaces 属性已弃用。
        空间应该在 action_space() 和 observation_space() 方法中定义。
        如果这些方法没有被重写，空间将从 self.observation_spaces/action_spaces 推断，并发出警告。

        这些属性在初始化后不应该改变。
        """
        self.possible_agents = ["player_" + str(r) for r in range(2)]

        # 可选：智能体名称和 ID 之间的映射
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    # 观察空间应该在这里定义。
    # lru_cache 允许观察和动作空间被记忆化，减少获取每个智能体空间所需的时钟周期。
    # 如果你的空间随时间变化，请删除这一行（禁用缓存）。
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium 空间在这里定义和记录：https://gymnasium.farama.org/api/spaces/
        # Discrete(4) 表示范围在 range(0, 4) 内的整数
        return Discrete(4)

    # 动作空间应该在这里定义。
    # 如果你的空间随时间变化，请删除这一行（禁用缓存）。
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        """
        渲染环境。在 human 模式下，它可以打印到终端，
        打开图形窗口，或打开其他人类可以看到和理解的显示。
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "你在调用 render 方法时没有指定任何渲染模式。"
            )
            return

        if len(self.agents) == 2:
            string = "当前状态：智能体1：{} ，智能体2：{}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "游戏结束"
        print(string)

    def close(self):
        """
        close 应该释放任何图形显示、子进程、网络连接
        或任何其他在用户不再使用环境时不应该保留的环境数据。
        """
        pass

    def reset(self, seed=None, options=None):
        """
        reset 需要初始化 `agents` 属性，并且必须设置环境，
        使得 render() 和 step() 可以无问题地调用。
        这里它初始化了 `num_moves` 变量，用于计算已经玩了多少局。
        返回每个智能体的观察。
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        # 即使只有一个值，观察也应该是 numpy 数组
        observations = {agent: np.array(NO_MOVE) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) 接收每个智能体的动作，并应该返回：
        - observations（观察）
        - rewards（奖励）
        - terminations（终止状态）
        - truncations（截断状态）
        - infos（信息）
        这些字典的格式都是 {agent_1: item_1, agent_2: item_2}
        """
        # 如果用户传入的动作没有智能体，则返回空的观察等
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # 所有智能体的奖励都放在要返回的 rewards 字典中
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = REWARD_MAP[
            (actions[self.agents[0]], actions[self.agents[1]])
        ]

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # 当前观察就是另一个玩家最近的动作
        # 这被转换为 int 类型的 numpy 值，以匹配我们在
        # observation_space() 中声明的类型
        observations = {
            self.agents[i]: np.array(actions[self.agents[1 - i]], dtype=np.int64)
            for i in range(len(self.agents))
        }
        self.state = observations

        # 通常 infos 中不会有任何信息，但必须
        # 为每个智能体都有一个条目
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos
