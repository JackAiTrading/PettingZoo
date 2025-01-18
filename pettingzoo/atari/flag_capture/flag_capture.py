# noqa: D212, D415
"""
夺旗游戏环境。

这个模块实现了一个双人夺旗游戏，玩家需要在迷宫中寻找并夺取对方的旗帜。
游戏基于 Atari 2600 的 Flag Capture 游戏。

# Flag Capture Game

```{figure} atari_flag_capture.gif
:width: 140px
:name: flag_capture
```

这个环境是 Atari 环境的一部分。请先阅读 Atari 环境的文档。

| Import                 | `from pettingzoo.atari import flag_capture_v2` |
|----------------------|------------------------------------------------|
| Action Space          | Discrete                                         |
| Parallel API          | Yes                                             |
| Manual Control        | No                                              |
| Agents                | `agents= ['first_0', 'second_0']`                |
| Agents Number         | 2                                               |
| Action Shape          | (1,)                                            |
| Action Values         | [0,9]                                           |
| Observation Shape    | (210, 160, 3)                                   |
| Observation Values   | (0,255)                                         |

这个游戏需要记忆和信息。

一个旗帜被隐藏在地图上。
你可以在地图上移动并检查
方块。如果你找到旗帜，
你将获得一分（你的对手将失去一分）。
如果是炸弹，你将被送回起始位置。
否则，它将给你一个关于旗帜位置的提示，
可能是方向或距离。
你的玩家需要能够使用自己的搜索和对手的搜索来快速有效地缩小旗帜的位置。

[官方 Flag Capture 游戏手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=183)


#### Environment Parameters

环境参数是所有 Atari 环境共有的，并在 [Base Atari 文档](../atari) 中描述。

### Action Space (Minimal)

在任何给定的回合中，代理可以选择 10 个动作。

| Action     | Behavior    |
|:---------:|-----------|
| 0         | No operation |
| 1         | Fire        |
| 2         | Move up     |
| 3         | Move right  |
| 4         | Move left   |
| 5         | Move down   |
| 6         | Move up-right|
| 7         | Move up-left|
| 8         | Move down-right|
| 9         | Move down-left|

### Version History

* v2: 最小化动作空间 (1.18.0)
* v1: 主要 API 更改 (1.4.0)
* v0: 初始版本发布 (1.0.0)

"""

import os
from glob import glob

from pettingzoo.atari.base_atari_env import (
    BaseAtariEnv,
    base_env_wrapper_fn,
    parallel_wrapper_fn,
)


def raw_env(**kwargs):
    """夺旗游戏环境的原始实现。

    参数:
        **kwargs: 传递给父类的参数

    返回:
        BaseAtariEnv: 夺旗游戏环境
    """
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="flag_capture", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
