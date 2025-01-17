# noqa: D212, D415
"""
# 拳击（Boxing）

```{figure} atari_boxing.gif
:width: 140px
:name: boxing
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import boxing_v2` |
|--------------------|------------------------------------------|
| 动作类型           | 离散                                     |
| 并行 API          | 支持                                      |
| 手动控制          | 不支持                                    |
| 智能体            | `agents= ['first_0', 'second_0']`        |
| 智能体数量        | 2                                        |
| 动作形状          | (1,)                                     |
| 动作值范围        | [0,17]                                   |
| 观察形状          | (210, 160, 3)                            |
| 观察值范围        | (0,255)                                  |


*拳击*是一个对抗性游戏，精确的控制和
对对手的适当反应是关键。

玩家有两分钟（约 1200 步）的时间在拳击台上
决一胜负。每一步，他们都可以移动和出拳。
成功的出拳可以得分，
远距离刺拳得 1 分，近距离重拳得 2 分，
击倒（KO）得 100 分（同时也会结束游戏）。
每当你得分时，你会获得相应分数的奖励，
而你的对手会受到相应分数的惩罚。

[官方拳击游戏手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=45)

#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

### 动作空间

在任何给定回合中，智能体可以从 18 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 出拳    |
| 2         | 向上移动 |
| 3         | 向右移动 |
| 4         | 向左移动 |
| 5         | 向下移动 |
| 6         | 向右上移动 |
| 7         | 向左上移动 |
| 8         | 向右下移动 |
| 9         | 向左下移动 |
| 10        | 向上出拳 |
| 11        | 向右出拳 |
| 12        | 向左出拳 |
| 13        | 向下出拳 |
| 14        | 向右上出拳 |
| 15        | 向左上出拳 |
| 16        | 向右下出拳 |
| 17        | 向左下出拳 |



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
        game="boxing", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
