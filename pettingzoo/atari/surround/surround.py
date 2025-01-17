# noqa: D212, D415
"""
# 包围（Surround）

```{figure} atari_surround.gif
:width: 140px
:name: surround
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import surround_v2` |
|----------------------|--------------------------------------------|
| 动作类型           | 离散                                       |
| 并行 API          | 支持                                        |
| 手动控制          | 不支持                                     |
| 智能体            | `agents= ['first_0', 'second_0']`          |
| 智能体数量        | 2                                          |
| 动作形状          | (1,)                                       |
| 动作值范围        | [0,4]                                      |
| 观察形状          | (210, 160, 3)                              |
| 观察值范围        | (0,255)                                    |


这是一个需要规划和策略的竞争性游戏。

在包围游戏中，你的目标是避开墙壁。如果你撞到墙壁，你会受到 -1 分的惩罚，而你的对手会得到 +1 分。

但两个玩家都会在身后留下一道墙壁，慢慢地让屏幕充满障碍物。为了尽可能长时间地避开障碍物，你必须规划路径以节省空间。一旦掌握了这一点，游戏的更高层面就会出现，两个玩家会字面意义上地尝试
用墙壁包围对方，这样对手就会用尽空间，被迫撞到墙上。

[官方包围游戏手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=943)

#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

### 动作空间（最小）

在任何给定回合中，智能体可以从 6 个动作中选择一个。（开火是虚拟动作，但为了保持连续编号）

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 开火（虚拟） |
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


def raw_env(**kwargs):
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="surround", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
