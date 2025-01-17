# noqa: D212, D415
"""
# 坦克大战（Combat: Tank）

```{figure} atari_combat_tank.gif
:width: 140px
:name: combat_tank
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import combat_tank_v3`     |
|--------------------|---------------------------------------------------|
| 动作类型           | 离散                                              |
| 并行 API          | 支持                                               |
| 手动控制          | 不支持                                             |
| 智能体            | `agents= ['first_0', 'second_0']`                 |
| 智能体数量        | 2                                                 |
| 动作形状          | (1,)                                              |
| 动作值范围        | [0,5]                                             |
| 观察形状          | (210, 160, 3)                                     |
| 观察值范围        | (0,255)                                           |


*坦克大战*的经典坦克模式是一个对抗性游戏，预判和位置选择是关键。

玩家在地图上移动。当你的子弹击中对手时，
你得一分。注意，当对手被击中时，会被炸飞穿过障碍物，这可能会让对手处于有利位置反击你。

每当你得分时，你获得 +1 奖励，你的对手受到 -1 惩罚。

[官方坦克大战手册](https://atariage.com/manual_html_page.php?SoftwareID=935)


#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

combat-tank 特有的参数如下：

``` python
combat_tank_v2.env(has_maze=True, is_invisible=False, billiard_hit=True)
```

`has_maze`：设置为 true 时，地图将是迷宫而不是开放场地

`is_invisible`：如果为 true，坦克在开火或撞墙时才可见。

`billiard_hit`：如果为 true，子弹会从墙壁反弹，实际上，就像台球一样，只有在子弹从墙壁反弹后击中对手的坦克才算得分。

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

* v2：最小动作空间 (1.18.0)
* v1：对整个 API 进行重大更改 (1.4.0)
* v0：初始版本发布 (1.0.0)


"""

import os
import warnings
from glob import glob

from pettingzoo.atari.base_atari_env import (
    BaseAtariEnv,
    base_env_wrapper_fn,
    parallel_wrapper_fn,
)


def raw_env(has_maze=True, is_invisible=False, billiard_hit=True, **kwargs):
    if has_maze is False and is_invisible is False and billiard_hit is False:
        warnings.warn(
            "坦克大战有一些有趣的参数可以考虑覆盖，包括 is_invisible（隐身）、billiard_hit（台球式击打）和 has_maze（迷宫）"
        )
    start_mapping = {
        (False, False): 1,
        (False, True): 8,
        (True, False): 10,
        (True, True): 13,
    }
    mode = start_mapping[(is_invisible, billiard_hit)] + has_maze
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="combat", num_players=2, mode_num=mode, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
