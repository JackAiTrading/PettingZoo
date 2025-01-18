import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
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


class raw_env(AECEnv):
    """
    metadata 保存环境常量。从 gymnasium 中，我们继承了 "render_modes" metadata，
    它指定了哪些模式可以传入 render() 方法。
    至少应该支持 human 模式。
    "name" metadata 允许环境以美观的方式打印。
    """

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

        # 可选：我们可以在这里将观察和动作空间定义为属性，以在相应的方法中使用
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    # 观察空间应该在这里定义。
    # lru_cache 允许观察和动作空间被记忆化，减少获取每个智能体空间所需的时钟周期。
    # 如果你的空间随时间变化，请删除这一行（禁用缓存）。
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium 空间在这里定义和记录：https://gymnasium.farama.org/api/spaces/
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

    def observe(self, agent):
        """
        observe 应该返回指定智能体的观察。这个函数
        应该在调用 reset() 后的任何时间返回一个合理的观察
        （虽然不一定是最新的可能观察）。
        """
        # 一个智能体的观察是另一个智能体的前一个状态
        return np.array(self.observations[agent])

    def close(self):
        """
        close 应该释放任何图形显示、子进程、网络连接
        或任何其他在用户不再使用环境时不应该保留的环境数据。
        """
        pass

    def reset(self, seed=None, options=None):
        """
        reset 需要初始化以下属性：
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        并且必须设置环境，使得 render()、step() 和 observe()
        可以无问题地调用。
        这里它设置了 state 字典（由 step() 使用）和 observations 字典（由 step() 和 observe() 使用）
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0
        """
        我们的 AgentSelector 工具允许轻松地循环遍历智能体列表。
        """
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) 接收当前智能体（由 agent_selection 指定）的动作，
        并需要更新：
        - rewards
        - _cumulative_rewards（累积奖励）
        - terminations
        - truncations
        - infos
        - agent_selection（到下一个智能体）
        以及 observe() 或 render() 使用的任何内部状态
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # 处理已经死亡的智能体的步骤
            # 接受该智能体的 None 动作，并将 agent_selection 移动到
            # 下一个死亡的智能体，或者如果没有更多死亡的智能体，则移动到下一个存活的智能体
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # 上一步行动的智能体的 _cumulative_rewards 已经被计算
        # （因为它是由 last() 返回的），所以这个
        # 智能体的 _cumulative_rewards 应该重新从 0 开始
        self._cumulative_rewards[agent] = 0

        # 存储当前智能体的动作
        self.state[self.agent_selection] = action

        # 如果是最后一个行动的智能体，则收集奖励
        if self._agent_selector.is_last():
            # 所有智能体的奖励都放在 .rewards 字典中
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            # truncations 字典必须为所有玩家更新。
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            # 观察当前状态
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            # 必要的，以便 observe() 在任何时候都返回合理的观察。
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # 在两个玩家都给出动作之前不分配奖励
            self._clear_rewards()

        # 选择下一个智能体。
        self.agent_selection = self._agent_selector.next()
        # 将 .rewards 添加到 ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
