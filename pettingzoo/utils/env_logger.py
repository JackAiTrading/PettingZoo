"""
环境日志记录模块。

这个模块提供了环境日志记录功能，用于记录和警告环境中的异常情况。
主要用于调试和监控环境的行为。

主要功能：
1. 记录环境警告信息
2. 提供不同级别的日志记录
3. 支持自定义警告消息
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import Any

import gymnasium.spaces


class EnvLogger:
    """环境日志记录器类。

    这个类提供了环境日志记录的静态方法，用于记录不同类型的警告和错误。

    类方法：
        warn_action_out_of_bound: 记录动作超出范围的警告
        warn_on_illegal_move: 记录非法移动的警告
        error_observe_before_reset: 记录在重置前观察的错误
        error_step_before_reset: 记录在重置前步进的错误
        warn_step_after_terminated_truncated: 记录在终止或截断后步进的警告
        error_render_before_reset: 记录在重置前渲染的错误
        error_agent_iter_before_reset: 记录在重置前迭代智能体的错误
        error_nan_action: 记录动作为NaN的错误
        error_state_before_reset: 记录在重置前获取状态的错误
    """

    mqueue: list[Any] = []
    _output: bool = True

    @staticmethod
    def get_logger() -> Logger:
        """获取日志记录器实例。

        返回:
            logging.Logger: 日志记录器实例
        """
        logger = logging.getLogger(__name__)
        return logger

    @staticmethod
    def _generic_warning(msg: Any) -> None:
        """输出警告消息。

        参数:
            msg (Any): 警告消息
        """
        logger = EnvLogger.get_logger()
        if not logger.hasHandlers():
            handler = EnvWarningHandler(mqueue=EnvLogger.mqueue)
            logger.addHandler(handler)
        logger.warning(msg)
        # needed to get the pytest runner to work correctly, and doesn't seem to have serious issues
        EnvLogger.mqueue.append(msg)

    @staticmethod
    def flush() -> None:
        """清空消息队列。"""
        EnvLogger.mqueue.clear()

    @staticmethod
    def suppress_output() -> None:
        """禁用日志记录。"""
        EnvLogger._output = False

    @staticmethod
    def unsuppress_output() -> None:
        """启用日志记录。"""
        EnvLogger._output = True

    @staticmethod
    def error_possible_agents_attribute_missing(name: str) -> None:
        """记录可能的智能体属性缺失错误。

        参数:
            name (str): 缺失属性的名称
        """
        raise AttributeError(
            f"[ERROR]: This environment does not support {name}. This means that either the environment has procedurally generated agents such that this property cannot be well defined (which requires special learning code to handle) or the environment was improperly configured by the developer."
        )

    @staticmethod
    def warn_action_out_of_bound(
        action: Any, action_space: gymnasium.spaces.Space, backup_policy: str
    ) -> None:
        """记录动作超出范围的警告。

        参数:
            action (Any): 执行的动作
            action_space (gymnasium.spaces.Space): 动作空间
            backup_policy (str): 备份策略
        """
        EnvLogger._generic_warning(
            f"[WARNING]: Received an action {action} that was outside action space {action_space}. Environment is {backup_policy}"
        )

    @staticmethod
    def warn_on_illegal_move() -> None:
        """记录非法移动的警告。"""
        EnvLogger._generic_warning(
            "[WARNING]: Illegal move made, game terminating with current player losing. \nobs['action_mask'] contains a mask of all legal moves that can be chosen."
        )

    @staticmethod
    def error_observe_before_reset() -> None:
        """记录在重置前观察的错误。"""
        assert False, "reset() needs to be called before observe."

    @staticmethod
    def error_step_before_reset() -> None:
        """记录在重置前步进的错误。"""
        assert False, "reset() needs to be called before step."

    @staticmethod
    def warn_step_after_terminated_truncated() -> None:
        """记录在终止或截断后步进的警告。"""
        EnvLogger._generic_warning(
            "[WARNING]: step() called after all agents are terminated or truncated. Should reset() first."
        )

    @staticmethod
    def error_render_before_reset() -> None:
        """记录在重置前渲染的错误。"""
        assert False, "reset() needs to be called before render."

    @staticmethod
    def error_agent_iter_before_reset() -> None:
        """记录在重置前迭代智能体的错误。"""
        assert False, "reset() needs to be called before agent_iter()."

    @staticmethod
    def error_nan_action() -> None:
        """记录动作为NaN的错误。"""
        assert False, "step() cannot take in a nan action."

    @staticmethod
    def error_state_before_reset() -> None:
        """记录在重置前获取状态的错误。"""
        assert False, "reset() needs to be called before state."


class EnvWarningHandler(logging.Handler):
    def __init__(self, *args, mqueue, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        self.mqueue = mqueue

    def emit(self, record: logging.LogRecord):
        m = self.format(record).rstrip("\n")
        self.mqueue.append(m)
        if EnvLogger._output:
            print(m)
