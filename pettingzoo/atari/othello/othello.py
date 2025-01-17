# noqa: D212, D415
"""
# 奥赛罗（Othello）

```{figure} atari_othello.gif
:width: 140px
:name: othello
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import othello_v3` |
|----------------------|-------------------------------------------|
| 动作类型           | 离散                                      |
| 并行 API          | 支持                                       |
| 手动控制          | 不支持                                    |
| 智能体            | `agents= ['first_0', 'second_0']`         |
| 智能体数量        | 2                                         |
| 动作形状          | (1,)                                      |
| 动作值范围        | [0,9]                                     |
| 观察形状          | (210, 160, 3)                             |
| 观察值范围        | (0,255)                                   |


经典的长期策略棋盘游戏。

目标是翻转对手的棋子。你可以通过在一行或对角线上放置一个棋子来翻转对手的棋子（将它们变成你的颜色），这样可以将对手的棋子夹在你自己的棋子之间。每回合你必须至少翻转一个棋子
（[奥赛罗规则](https://www.mastersofgames.com/rules/reversi-othello-rules.htm)）。

注意，众所周知，在任何时候最大化己方棋子数量的贪婪策略是一个非常糟糕的策略，这使得学习变得更有趣。

要放置一个棋子，必须将光标移动到棋盘上的有效位置并按下开火键。控制有一定的延迟，这意味着需要重复一段时间的动作才能生效。

分数是你在棋盘上的棋子数量。给出的奖励是相对奖励，所以如果你在一回合中翻转了对手的 5 个棋子，你会得到 +6 奖励，你的对手会得到 -6 奖励，因为你有 6 个新棋子（你放置的 1 个加上你翻转的 5 个）。

注意，贪婪地追求这种奖励是一个糟糕的长期策略，所以为了成功解决这个游戏，你必须进行长期思考。

当一个玩家无法移动时，双方的棋子会被计数，拥有最多棋子的玩家获胜！（获得 +1 奖励，对手获得 -1 奖励）。

这是一个计时游戏：如果一个玩家在 10 秒后还没有行动，那么该玩家将被扣除 1 分，对手不会得到任何奖励，计时器重置。这可以防止一个玩家无限期地拖延游戏，但也意味着这不再是一个纯零和游戏。

[官方奥赛罗手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=335)

#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

### 动作空间（最小）

在任何给定回合中，智能体可以从 10 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 开火    |
| 2         | 向上移动 |
| 3         | 向右移动 |
| 4         | 向左移动 |
| 5         | 向下移动 |
| 6         | 向右上移动 |
| 7         | 向左上移动 |
| 8         | 向右下移动 |
| 9         | 向左下移动 |

### 版本历史

* v3：最小动作空间 (1.18.0)
* v2：对整个 API 进行重大更改 (1.4.0)
* v1：修复了奥赛罗自动重置问题 (1.2.1)
* v0：初始版本发布 (1.0.0)


"""

import os
from glob import glob

from pettingzoo.atari.base_atari_env import (
    BaseAtariEnv,
    base_env_wrapper_fn,
    parallel_wrapper_fn,
)


def raw_env(**kwargs):
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="othello", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
