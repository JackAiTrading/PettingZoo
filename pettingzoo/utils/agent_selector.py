"""
智能体选择器模块。

这个模块提供了在AEC环境中选择下一个活跃智能体的功能。
主要用于管理智能体的轮转顺序和选择逻辑。

主要功能：
1. 管理智能体的轮转顺序
2. 提供智能体选择机制
3. 支持智能体的重置和更新
"""

from __future__ import annotations

from typing import Any, List, Optional
from warnings import warn


class AgentSelector:
    """智能体选择器类。

    这个类用于在AEC环境中管理智能体的选择顺序。
    它维护一个智能体列表，并提供方法来选择下一个活跃智能体。

    属性：
        agent_order (list): 智能体的顺序列表
        _current_agent (int): 当前选中的智能体索引
        selected_agent (Any): 当前选中的智能体

    示例：
        >>> from pettingzoo.utils import AgentSelector
        >>> agent_selector = AgentSelector(agent_order=["player1", "player2"])
        >>> agent_selector.reset()
        'player1'
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        True
        >>> agent_selector.reinit(agent_order=["player2", "player1"])
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        False
    """

    def __init__(self, agent_order: List[Any]):
        """初始化智能体选择器。

        参数：
            agent_order (List[Any]): 智能体的顺序列表
        """
        self.reinit(agent_order)

    def reinit(self, agent_order: List[Any]) -> None:
        """重置智能体选择器。

        参数：
            agent_order (List[Any]): 智能体的顺序列表
        """
        self.agent_order = agent_order
        self._current_agent = 0
        self.selected_agent = 0

    def reset(self) -> Any:
        """重置选择器状态。

        返回：
            Any: 第一个智能体的名称
        """
        self.reinit(self.agent_order)
        return self.next()

    def next(self) -> Any:
        """选择下一个智能体。

        返回：
            Any: 下一个智能体的名称

        注意：
            如果到达列表末尾，会自动回到开始位置
        """
        self._current_agent = (self._current_agent + 1) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent

    def is_last(self) -> bool:
        """检查当前是否是最后一个智能体。

        返回：
            bool: 如果当前是最后一个智能体则返回True，否则返回False
        """
        return self.selected_agent == self.agent_order[-1]

    def is_first(self) -> bool:
        """检查当前是否是第一个智能体。

        返回：
            bool: 如果当前是第一个智能体则返回True，否则返回False
        """
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other: AgentSelector) -> bool:
        """比较两个选择器是否相等。

        参数：
            other (AgentSelector): 要比较的另一个对象

        返回：
            bool: 如果两个选择器相等则返回True，否则返回False
        """
        if not isinstance(other, AgentSelector):
            return NotImplemented

        return (
            self.agent_order == other.agent_order
            and self._current_agent == other._current_agent
            and self.selected_agent == other.selected_agent
        )


class agent_selector(AgentSelector):
    """Deprecated version of AgentSelector. Use that instead."""

    def __init__(self, *args, **kwargs):
        warn(
            "agent_selector is deprecated, please use AgentSelector",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
