# noqa: D212, D415
"""
# 篮球乒乓球（Basketball Pong）

```{figure} atari_basketball_pong.gif
:width: 140px
:name: basketball_pong
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import basketball_pong_v3` |
|--------------------|---------------------------------------------------|
| 动作类型           | 离散                                              |
| 并行 API          | 支持                                              |
| 手动控制          | 不支持                                            |
| 智能体            | `agents= ['first_0', 'second_0']`                 |
| 智能体数量        | 2                                                 |
| 动作形状          | (1,)                                              |
| 动作值范围        | [0,5]                                             |
| 观察形状          | (210, 160, 3)                                     |
| 观察值范围        | (0,255)                                           |


这是一个竞争性的控制游戏。

尝试将球投入对手的篮筐。但你不能移动到对方的半场。得分会给你的对手 -1 分奖励。

发球是有时间限制的：如果玩家在收到球后 2 秒内没有发球，他们将获得 -1 分，计时器重置。这可以防止一名玩家无限期地拖延游戏，但也意味着这不再是一个纯零和游戏。


[官方 Video Olympics 手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

Basketball_Pong 特有的参数如下：

``` python
basketball_pong_v3.env(num_players=2)
```

`num_players`：玩家数量（必须是 2 或 4）

### 动作空间（最小）

在任何给定回合中，智能体可以从 6 个动作中选择一个。
| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 发射    |
| 2         | 向上移动 |
| 3         | 向右移动 |
| 4         | 向左移动 |
| 5         | 向下移动 |

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


def raw_env(num_players=2, **kwargs):
    assert num_players == 2 or num_players == 4, "乒乓球游戏只支持 2 或 4 名玩家"
    mode_mapping = {2: 45, 4: 49}
    mode = mode_mapping[num_players]
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
