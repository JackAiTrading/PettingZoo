"""
环境转换工具模块。

这个模块提供了在不同类型环境之间进行转换的功能。
主要支持AEC（Agent Environment Cycle）环境和并行环境之间的相互转换。

主要功能：
1. 将并行环境转换为AEC环境
2. 将AEC环境转换为并行环境
3. 提供转换后环境的包装器类
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from pettingzoo.utils.env import AECEnv, ParallelEnv


def parallel_to_aec(parallel_env: ParallelEnv) -> AECEnv:
    """将并行环境转换为AEC环境。

    参数:
        parallel_env (ParallelEnv): 要转换的并行环境

    返回:
        AECEnv: 转换后的AEC环境

    示例:
        >>> parallel_env = YourParallelEnv()
        >>> aec_env = parallel_to_aec(parallel_env)
    """
    return parallel_wrapper_fn(parallel_env)


def aec_to_parallel(aec_env: AECEnv) -> ParallelEnv:
    """将AEC环境转换为并行环境。

    参数:
        aec_env (AECEnv): 要转换的AEC环境

    返回:
        ParallelEnv: 转换后的并行环境

    示例:
        >>> aec_env = YourAECEnv()
        >>> parallel_env = aec_to_parallel(aec_env)
    """
    return aec_wrapper_fn(aec_env)


def parallel_wrapper_fn(env: ParallelEnv) -> AECEnv:
    """创建并行环境到AEC环境的包装器。

    参数:
        env (ParallelEnv): 要包装的并行环境

    返回:
        AECEnv: 包装后的AEC环境
    """
    return ParallelToAECWrapper(env)


def aec_wrapper_fn(env: AECEnv) -> ParallelEnv:
    """创建AEC环境到并行环境的包装器。

    参数:
        env (AECEnv): 要包装的AEC环境

    返回:
        ParallelEnv: 包装后的并行环境
    """
    return AECToParallelWrapper(env)


class ParallelToAECWrapper(AECEnv):
    """将并行环境包装为AEC环境的包装器类。

    这个类将并行环境转换为AEC环境，使其能够按照智能体循环的方式运行。

    属性:
        env (ParallelEnv): 被包装的并行环境
        agents (list): 当前活跃的智能体列表
        possible_agents (list): 所有可能的智能体列表
        agent_selection (str): 当前选中的智能体
        action_spaces (dict): 每个智能体的动作空间
        observation_spaces (dict): 每个智能体的观察空间
    """

    def __init__(self, parallel_env: ParallelEnv):
        """初始化包装器。

        参数:
            parallel_env (ParallelEnv): 要包装的并行环境
        """
        self.env = parallel_env
        self.metadata = parallel_env.metadata

        # 初始化环境属性
        self.agents = self.env.agents[:]
        self.possible_agents = self.env.possible_agents[:]
        self.agent_selection = self.agents[0]

        # 初始化空间
        self.action_spaces = {agent: space for agent, space in self.env.action_spaces.items()}
        self.observation_spaces = {agent: space for agent, space in self.env.observation_spaces.items()}

        # 初始化状态
        self._observations = {agent: None for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._actions = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """重置环境到初始状态。

        参数:
            seed (Optional[int]): 随机数种子
            options (Optional[dict]): 重置选项
        """
        self._observations, self.infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self.agent_selection = self.agents[0]
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._actions = {}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}

    def observe(self, agent: str) -> Union[np.ndarray, dict]:
        """获取指定智能体的观察。

        参数:
            agent (str): 智能体名称

        返回:
            Union[np.ndarray, dict]: 智能体的观察
        """
        return self._observations[agent]

    def step(self, action: Any) -> None:
        """执行一步环境交互。

        参数:
            action: 当前智能体的动作
        """
        agent = self.agent_selection
        self._actions[agent] = action

        if len(self._actions) == len(self.agents):
            observations, rewards, terminations, truncations, infos = self.env.step(self._actions)
            self._observations = observations
            self.rewards = rewards
            self.terminations = terminations
            self.truncations = truncations
            self.infos = infos
            self.agents = self.env.agents[:]
            self._actions = {}
            self.agent_selection = self.agents[0]
        else:
            self.agent_selection = self.agents[len(self._actions)]

    def render(self) -> Optional[Union[np.ndarray, str, list]]:
        """渲染环境。

        返回:
            Optional[Union[np.ndarray, str, list]]: 渲染结果
        """
        return self.env.render()

    def state(self) -> np.ndarray:
        """获取环境的全局状态。

        返回:
            np.ndarray: 环境的全局状态
        """
        return self.env.state()

    def close(self) -> None:
        """关闭环境。"""
        self.env.close()


class AECToParallelWrapper(ParallelEnv):
    """将AEC环境包装为并行环境的包装器类。

    这个类将AEC环境转换为并行环境，使其能够同时处理所有智能体的动作。

    属性:
        env (AECEnv): 被包装的AEC环境
        agents (list): 当前活跃的智能体列表
        possible_agents (list): 所有可能的智能体列表
        action_spaces (dict): 每个智能体的动作空间
        observation_spaces (dict): 每个智能体的观察空间
    """

    def __init__(self, aec_env: AECEnv):
        """初始化包装器。

        参数:
            aec_env (AECEnv): 要包装的AEC环境
        """
        self.env = aec_env
        self.metadata = aec_env.metadata

        # 初始化环境属性
        self.agents = self.env.agents[:]
        self.possible_agents = self.env.possible_agents[:]

        # 初始化空间
        self.action_spaces = dict(self.env.action_spaces)
        self.observation_spaces = dict(self.env.observation_spaces)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """重置环境到初始状态。

        参数:
            seed (Optional[int]): 随机数种子
            options (Optional[dict]): 重置选项

        返回:
            Tuple[Dict, Dict]: (observations, infos)
        """
        self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        observations = {agent: self.env.observe(agent) for agent in self.agents}
        infos = self.env.infos
        return observations, infos

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """执行一步环境交互。

        参数:
            actions (Dict[str, Any]): 每个智能体的动作

        返回:
            Tuple[Dict, Dict, Dict, Dict, Dict]: (observations, rewards, terminations, truncations, infos)
        """
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            rewards[agent] = reward
            terminations[agent] = termination
            truncations[agent] = truncation
            infos[agent] = info

            if termination or truncation:
                action = None
            else:
                action = actions[agent]
            self.env.step(action)

        observations = {agent: self.env.observe(agent) for agent in self.agents}
        self.agents = self.env.agents
        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[Union[np.ndarray, str, list]]:
        """渲染环境。

        返回:
            Optional[Union[np.ndarray, str, list]]: 渲染结果
        """
        return self.env.render()

    def state(self) -> np.ndarray:
        """获取环境的全局状态。

        返回:
            np.ndarray: 环境的全局状态
        """
        return self.env.state()

    def close(self) -> None:
        """关闭环境。"""
        self.env.close()
