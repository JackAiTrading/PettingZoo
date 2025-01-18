"""
环境基类模块。

这个模块定义了PettingZoo中所有环境的基类。包括：
1. AEC（Agent Environment Cycle）环境基类
2. 并行环境基类
3. 环境验证工具

主要功能：
1. 定义环境接口规范
2. 提供基础的环境操作方法
3. 实现通用的环境状态管理
"""


from __future__ import annotations

import warnings
from typing import Any, Dict, Generic, Iterable, Iterator, TypeVar

import gymnasium.spaces
import numpy as np

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

# deprecated
ObsDict = Dict[AgentID, ObsType]

# deprecated
ActionDict = Dict[AgentID, ActionType]

"""
Base environment definitions

See docs/api.md for api documentation
See docs/dev_docs.md for additional documentation and an example environment.
"""


class AECEnv(Generic[AgentID, ObsType, ActionType]):
    """AEC（Agent Environment Cycle）环境基类。

    这个类定义了AEC环境的标准接口，所有AEC环境都应该继承这个类。
    AEC环境的特点是智能体轮流与环境交互，每次只有一个智能体可以行动。

    属性：
        agents (list): 当前活跃的智能体列表
        possible_agents (list): 所有可能的智能体列表
        agent_selection (str): 当前选中的智能体
        observation_spaces (dict): 每个智能体的观察空间
        action_spaces (dict): 每个智能体的动作空间
        rewards (dict): 每个智能体的即时奖励
        terminations (dict): 每个智能体的终止状态
        truncations (dict): 每个智能体的截断状态
        infos (dict): 每个智能体的额外信息
    """

    metadata: dict[str, Any]  # Metadata for the environment

    # All agents that may appear in the environment
    possible_agents: list[AgentID]
    agents: list[AgentID]  # Agents active at any given time

    observation_spaces: dict[
        AgentID, gymnasium.spaces.Space
    ]  # Observation space for each agent
    # Action space for each agent
    action_spaces: dict[AgentID, gymnasium.spaces.Space]

    # Whether each agent has just reached a terminal state
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    rewards: dict[AgentID, float]  # Reward from the last step for each agent
    # Cumulative rewards for each agent
    _cumulative_rewards: dict[AgentID, float]
    infos: dict[
        AgentID, dict[str, Any]
    ]  # Additional information from the last step for each agent

    agent_selection: AgentID  # The agent currently being stepped

    def __init__(self):
        """初始化AEC环境。"""
        pass

    def step(self, action: ActionType) -> None:
        """执行一步环境交互。

        参数：
            action: 当前智能体的动作

        注意：
            这个方法需要在子类中实现
        """
        raise NotImplementedError

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        """重置环境到初始状态。

        参数：
            seed (int, 可选): 随机数种子
            options (dict, 可选): 重置选项

        注意：
            这个方法需要在子类中实现
        """
        raise NotImplementedError

    # TODO: Remove `Optional` type below
    def observe(self, agent: AgentID) -> ObsType | None:
        """获取指定智能体的观察。

        参数：
            agent (str): 智能体名称

        返回：
            object: 智能体的观察

        注意：
            这个方法需要在子类中实现
        """
        raise NotImplementedError

    def render(self) -> None | np.ndarray | str | list:
        """渲染环境的当前状态。

        注意：
            这个方法需要在子类中实现
        """
        raise NotImplementedError

    def state(self) -> np.ndarray:
        """获取环境的全局状态。

        返回：
            object: 环境的全局状态

        注意：
            这个方法在子类中是可选的
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def close(self):
        """关闭环境，释放资源。

        注意：
            这个方法需要在子类中实现
        """
        pass

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """获取指定智能体的观察空间。

        参数：
            agent (str): 智能体名称

        返回：
            spaces.Space: 观察空间
        """
        warnings.warn(
            "Your environment should override the observation_space function. Attempting to use the observation_spaces dict attribute."
        )
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """获取指定智能体的动作空间。

        参数：
            agent (str): 智能体名称

        返回：
            spaces.Space: 动作空间
        """
        warnings.warn(
            "Your environment should override the action_space function. Attempting to use the action_spaces dict attribute."
        )
        return self.action_spaces[agent]

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def _deads_step_first(self) -> AgentID:
        """处理已终止智能体的动作。

        这个方法在智能体已经终止但仍然尝试执行动作时被调用。

        返回：
            AgentID: 当前智能体
        """
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        return self.agent_selection

    def _clear_rewards(self) -> None:
        """清除所有智能体的奖励。"""
        for agent in self.rewards:
            self.rewards[agent] = 0

    def _accumulate_rewards(self) -> None:
        """累积奖励。

        这个方法用于累积每个智能体的奖励。
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

    def agent_iter(self, max_iter: int = 2**63) -> AECIterable:
        """获取智能体迭代器。

        参数：
            max_iter (int, 可选): 最大迭代次数

        返回：
            AECIterable: 智能体迭代器
        """
        return AECIterable(self, max_iter)

    def last(
        self, observe: bool = True
    ) -> tuple[ObsType | None, float, bool, bool, dict[str, Any]]:
        """获取当前智能体的状态。

        参数：
            observe (bool, 可选): 是否观察当前智能体

        返回：
            tuple: (observation, cumulative_reward, terminated, truncated, info)
                - observation (object): 当前智能体的观察
                - cumulative_reward (float): 当前智能体的累积奖励
                - terminated (bool): 当前智能体是否终止
                - truncated (bool): 当前智能体是否截断
                - info (dict): 当前智能体的额外信息
        """
        agent = self.agent_selection
        assert agent is not None
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def _was_dead_step(self, action: ActionType) -> None:
        """处理已终止智能体的动作。

        这个方法在智能体已经终止但仍然尝试执行动作时被调用。

        参数：
            action: 尝试执行的动作
        """
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
            self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                assert self._skip_agent_selection is not None
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

    def __str__(self) -> str:
        """获取环境名称。

        返回：
            str: 环境名称
        """
        if hasattr(self, "metadata"):
            return self.metadata.get("name", self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self) -> AECEnv[AgentID, ObsType, ActionType]:
        return self


class AECIterable(Iterable[AgentID], Generic[AgentID, ObsType, ActionType]):
    def __init__(self, env, max_iter):
        self.env = env
        self.max_iter = max_iter

    def __iter__(self) -> AECIterator[AgentID, ObsType, ActionType]:
        return AECIterator(self.env, self.max_iter)


class AECIterator(Iterator[AgentID], Generic[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], max_iter: int):
        self.env = env
        self.iters_til_term = max_iter

    def __next__(self) -> AgentID:
        if not self.env.agents or self.iters_til_term <= 0:
            raise StopIteration
        self.iters_til_term -= 1
        return self.env.agent_selection

    def __iter__(self) -> AECIterator[AgentID, ObsType, ActionType]:
        return self


class ParallelEnv(Generic[AgentID, ObsType, ActionType]):
    """并行环境基类。

    这个类定义了并行环境的标准接口，所有并行环境都应该继承这个类。
    并行环境的特点是所有智能体可以同时行动。

    属性：
        agents (list): 当前活跃的智能体列表
        possible_agents (list): 所有可能的智能体列表
        observation_spaces (dict): 每个智能体的观察空间
        action_spaces (dict): 每个智能体的动作空间
    """

    metadata: dict[str, Any]

    agents: list[AgentID]
    possible_agents: list[AgentID]
    observation_spaces: dict[
        AgentID, gymnasium.spaces.Space
    ]  # Observation space for each agent
    action_spaces: dict[AgentID, gymnasium.spaces.Space]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """重置环境到初始状态。

        参数：
            seed (int, 可选): 随机数种子
            options (dict, 可选): 重置选项

        返回：
            tuple: (observations, infos)
                - observations (dict): 每个智能体的初始观察
                - infos (dict): 每个智能体的初始信息

        注意：
            这个方法需要在子类中实现
        """
        raise NotImplementedError

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """执行一步环境交互。

        参数：
            actions (dict): 每个智能体的动作

        返回：
            tuple: (observations, rewards, terminations, truncations, infos)
                - observations (dict): 每个智能体的新观察
                - rewards (dict): 每个智能体的奖励
                - terminations (dict): 每个智能体的终止状态
                - truncations (dict): 每个智能体的截断状态
                - infos (dict): 每个智能体的额外信息

        注意：
            这个方法需要在子类中实现
        """
        raise NotImplementedError

    def render(self) -> None | np.ndarray | str | list:
        """渲染环境的当前状态。

        注意：
            这个方法需要在子类中实现
        """
        raise NotImplementedError

    def close(self):
        """关闭环境，释放资源。

        注意：
            这个方法需要在子类中实现
        """
        pass

    def state(self) -> np.ndarray:
        """获取环境的全局状态。

        返回：
            object: 环境的全局状态

        注意：
            这个方法在子类中是可选的
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """获取指定智能体的观察空间。

        参数：
            agent (str): 智能体名称

        返回：
            spaces.Space: 观察空间
        """
        warnings.warn(
            "Your environment should override the observation_space function. Attempting to use the observation_spaces dict attribute."
        )
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """获取指定智能体的动作空间。

        参数：
            agent (str): 智能体名称

        返回：
            spaces.Space: 动作空间
        """
        warnings.warn(
            "Your environment should override the action_space function. Attempting to use the action_spaces dict attribute."
        )
        return self.action_spaces[agent]

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def __str__(self) -> str:
        """获取环境名称。

        返回：
            str: 环境名称
        """
        if hasattr(self, "metadata"):
            return self.metadata.get("name", self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self) -> ParallelEnv:
        return self
