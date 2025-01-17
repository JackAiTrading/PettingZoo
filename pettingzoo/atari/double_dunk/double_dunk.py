# noqa: D212, D415
"""
# 双人扣篮（Double Dunk）

```{figure} atari_double_dunk.gif
:width: 140px
:name: double_dunk
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import double_dunk_v3` |
|--------------------|-----------------------------------------------|
| 动作类型           | 离散                                          |
| 并行 API          | 支持                                           |
| 手动控制          | 不支持                                         |
| 智能体            | agents= ['first_0', 'second_0']               |
| 智能体数量        | 2                                             |
| 动作形状          | (1,)                                          |
| 动作值范围        | [0,17]                                        |
| 观察形状          | (210, 160, 3)                                 |
| 观察值范围        | (0,255)                                       |


这是一个结合了控制和精确选择的对抗性游戏。

游戏分为两个阶段：选择和比赛。选择可能会
很困难，因为你必须保持同一个动作几步，然后
选择 0 动作。策略选择是有时间限制的：如果玩家在 2 秒（120 帧）后没有选择任何动作，
那么玩家将受到 -1 的惩罚，计时器重置。这可以防止一名玩家无限期地拖延游戏，但也意味着这不再是一个纯零和游戏。

一旦比赛开始，每队有两名球员。你一次只能控制
一名球员，你控制哪个球员取决于所选择的战术。
得分规则对篮球迷来说应该很熟悉（每次成功投篮得 2-3 分）。

[官方双人扣篮手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=153)


#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

### 动作空间

在任何给定回合中，智能体可以从 18 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 投球    |
| 2         | 向上移动 |
| 3         | 向右移动 |
| 4         | 向左移动 |
| 5         | 向下移动 |
| 6         | 向右上移动 |
| 7         | 向左上移动 |
| 8         | 向右下移动 |
| 9         | 向左下移动 |
| 10        | 向上投球 |
| 11        | 向右投球 |
| 12        | 向左投球 |
| 13        | 向下投球 |
| 14        | 向右上投球 |
| 15        | 向左上投球 |
| 16        | 向右下投球 |
| 17        | 向左下投球 |

### 版本历史

* v3：最小动作空间 (1.18.0)
* v2：取消动作计时器 (1.9.0)
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


def raw_env(**kwargs):
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="double_dunk", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
