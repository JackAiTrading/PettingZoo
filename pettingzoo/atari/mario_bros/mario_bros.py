# noqa: D212, D415
"""
# 马里奥兄弟（Mario Bros）

```{figure} atari_mario_bros.gif
:width: 140px
:name: mario_bros
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import mario_bros_v3` |
|----------------------|----------------------------------------------|
| 动作类型           | 离散                                         |
| 并行 API          | 支持                                          |
| 手动控制          | 不支持                                       |
| 智能体            | `agents= ['first_0', 'second_0']`            |
| 智能体数量        | 2                                            |
| 动作形状          | (1,)                                         |
| 动作值范围        | [0,17]                                       |
| 观察形状          | (210, 160, 3)                                |
| 观察值范围        | (0,255)                                      |


一个需要规划和控制的混合总和游戏。

主要目标是将害虫从地板上踢下去。这需要两个步骤：

1. 击打害虫下方的地板，将其翻转。这会使害虫仰面朝上。
2. 你需要移动到害虫所在的地板上，然后可以将其踢下去。这会获得 +800 奖励。

注意，由于这个过程有两个步骤，两个智能体有机会进行合作，通过互相帮助击倒害虫并收集它们（可能让双方都能更快获得奖励），或者智能体也可以窃取对方的工作成果。

如果你撞到一个活跃的害虫或火球，你会失去一条生命。如果你失去所有生命，你就结束了，而另一个玩家继续游戏。在获得 20000 分后，你可以获得一条新的生命。

还有其他获得分数的方式，比如收集奖励金币或薄饼，每个可获得 800 分。

[官方马里奥兄弟手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=286)

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
* v2：对整个 API 进行重大更改 (1.4.0)
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
        game="mario_bros", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
