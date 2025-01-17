# noqa: D212, D415
"""
# 活埋：合作版（Emtombed: Cooperative）

```{figure} atari_entombed_cooperative.gif
:width: 140px
:name: entombed_cooperative
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import entombed_cooperative_v3` |
|----------------------|--------------------------------------------------------|
| 动作类型           | 离散                                                   |
| 并行 API          | 支持                                                    |
| 手动控制          | 不支持                                                 |
| 智能体            | `agents= ['first_0', 'second_0']`                      |
| 智能体数量        | 2                                                      |
| 动作形状          | (1,)                                                   |
| 动作值范围        | [0,17]                                                 |
| 观察形状          | (210, 160, 3)                                          |
| 观察值范围        | (0,255)                                                |
| 平均总奖励        | 6.23                                                   |


活埋的合作版本是一个探索游戏，
你需要与队友合作，尽可能深入
迷宫。

你们两个都需要快速在一个不断生成的迷宫中导航，
而你只能看到其中的一部分。如果你被困住了，你就输了。
注意，你很容易发现自己陷入死胡同，只能通过使用稀有的能量提升道具来逃脱。
如果玩家通过使用这些能量提升道具互相帮助，他们可以坚持更久。注意，最佳协调要求智能体在地图的两侧，因为能量提升道具会出现在一侧或另一侧，但可以用来打破两侧的墙
（破坏是对称的，会影响屏幕的两半）。
此外，还有危险的僵尸潜伏在周围需要避开。

奖励的设计与单人游戏的奖励相同。具体来说，一个活埋关卡被分为 5 个不可见的区域。你在改变区域后或重置关卡后立即获得奖励。注意，这意味着当你失去一条生命时会获得奖励，
因为它会重置关卡，但当你失去最后一条生命时不会获得奖励，因为游戏会在关卡重置之前终止。


[官方活埋游戏手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=165)


#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

### 动作空间

在任何给定回合中，智能体可以从 18 个动作中选择一个。

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
| 10        | 向上开火 |
| 11        | 向右开火 |
| 12        | 向左开火 |
| 13        | 向下开火 |
| 14        | 向右上开火 |
| 15        | 向左上开火 |
| 16        | 向右下开火 |
| 17        | 向左下开火 |

### 版本历史

* v3：最小动作空间 (1.18.0)
* v2：对整个 API 进行重大更改，修复了活埋游戏的奖励 (1.4.0)
* v1：修复了所有环境处理过早死亡的方式 (1.3.0)
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
        game="entombed", num_players=2, mode_num=3, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
