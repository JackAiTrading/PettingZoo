# noqa: D212, D415
"""
# 太空侵略者（Space Invaders）

```{figure} atari_space_invaders.gif
:width: 140px
:name: space_invaders
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import space_invaders_v2` |
|----------------------|--------------------------------------------------|
| 动作类型           | 离散                                             |
| 并行 API          | 支持                                              |
| 手动控制          | 不支持                                           |
| 智能体            | `agents= ['first_0', 'second_0']`                |
| 智能体数量        | 2                                                |
| 动作形状          | (1,)                                             |
| 动作值范围        | [0,5]                                            |
| 观察形状          | (210, 160, 3)                                    |
| 观察值范围        | (0,255)                                          |


经典的 Atari 游戏，但有两艘由两个玩家控制的飞船，每个玩家都试图最大化他们的得分。

这个游戏具有合作性，玩家可以通过合作来通关以最大化他们的得分。普通外星人根据它们的起始高度可以得到 5-30 分，而在屏幕顶部飞过的飞船值 100 分。

然而，游戏也有竞争性的一面，当另一个玩家被外星人击中时，玩家会获得 200 分的奖励。所以破坏另一个玩家也是一种可能的策略。

飞船之间共享生命数，即当一艘飞船被击中 3 次时游戏结束。

[官方太空侵略者手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=460)

#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

太空侵略者特有的参数如下：

``` python
space_invaders_v2.env(alternating_control=False, moving_shields=True,
zigzaging_bombs=False, fast_bomb=False, invisible_invaders=False)
```

`alternating_control`：每次只有两个玩家中的一个有开火选项。如果你开火，你的对手就可以开火。但是，你不能永远保持开火能力，最终控制权会转移给你的对手。

`moving_shields`：护盾来回移动，提供的保护不太可靠。

`zigzaging_bombs`：入侵者的炸弹来回移动，更难避开。

`fast_bomb`：炸弹速度更快，更难避开。

`invisible_invaders`：入侵者是隐形的，更难击中。

### 动作空间（最小）

在任何给定回合中，智能体可以从 6 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 开火    |
| 2         | 向上移动 |
| 3         | 向右移动 |
| 4         | 向左移动 |
| 5         | 向下移动 |

### 版本历史

* v2：最小动作空间 (1.18.0)
* v1：对整个 API 进行重大更改 (1.4.0)
* v0：初始版本发布 (1.0.0)


"""

import os
from glob import glob

from pettingzoo.atari.base_atari_env import (
    BaseAtariEnv,
    base_env_wrapper_fn,
    parallel_wrapper_fn,
)


def raw_env(
    alternating_control=False,
    moving_shields=True,
    zigzaging_bombs=False,
    fast_bomb=False,
    invisible_invaders=False,
    **kwargs,
):
    mode = 33 + (
        moving_shields * 1
        + zigzaging_bombs * 2
        + fast_bomb * 4
        + invisible_invaders * 8
        + alternating_control * 16
    )
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="space_invaders", num_players=2, mode_num=mode, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
