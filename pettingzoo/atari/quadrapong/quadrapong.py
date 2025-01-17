# noqa: D212, D415
"""
# 四人乒乓（Quadrapong）

```{figure} atari_quadrapong.gif
:width: 140px
:name: quadrapong
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import quadrapong_v4`             |
|----------------------|----------------------------------------------------------|
| 动作类型           | 离散                                                     |
| 并行 API          | 支持                                                      |
| 手动控制          | 不支持                                                   |
| 智能体            | `agents= ['first_0', 'second_0', 'third_0', 'fourth_0']` |
| 智能体数量        | 4                                                        |
| 动作形状          | (1,)                                                     |
| 动作值范围        | [0,5]                                                    |
| 观察形状          | (210, 160, 3)                                            |
| 观察值范围        | (0,255)                                                  |


四人团队对战。

每个玩家控制一个球拍并防守一个得分区域。然而，这是一个团队游戏，所以 4 个得分区域中的两个属于同一个团队。因此，一个团队必须尝试协调，将球从他们的得分区域引向对手的得分区域。
具体来说，`first_0` 和 `third_0` 在一个队，`second_0` 和 `fourth_0` 在另一个队。

得分会给你的团队 +1 奖励，给对手团队 -1 惩罚。

发球是有时间限制的：如果玩家在收到球后 2 秒内没有发球，他们的团队会受到 -1 分的惩罚，计时器重置。这可以防止一个玩家无限期地拖延游戏，但也意味着这不再是一个纯零和游戏。


[官方视频奥运会手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

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

* v4：最小动作空间 (1.18.0)
* v3：取消动作计时器 (1.9.0)
* v1：对整个 API 进行重大更改 (1.4.0)
* v2：修复了四人乒乓的奖励 (1.2.0)
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
    mode = 33
    num_players = 4
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="pong", num_players=num_players, mode_num=mode, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
