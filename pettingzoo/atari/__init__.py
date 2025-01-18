"""Atari 环境的实现。

这些环境包括经典的 Atari 多人游戏。
"""

from pettingzoo.utils.deprecated_module import deprecated_handler


def __getattr__(env_name):
    return deprecated_handler(env_name, __path__, __name__)
