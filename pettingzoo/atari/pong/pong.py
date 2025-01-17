# noqa: D212, D415
"""
# 乒乓（Pong）

```{figure} atari_pong.gif
:width: 140px
:name: pong
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import pong_v3` |
|----------------------|----------------------------------------|
| 动作类型           | 离散                                   |
| 并行 API          | 支持                                    |
| 手动控制          | 不支持                                 |
| 智能体            | `agents= ['first_0', 'second_0']`      |
| 智能体数量        | 2                                      |
| 动作形状          | (1,)                                   |
| 动作值范围        | [0,5]                                  |
| 观察形状          | (210, 160, 3)                          |
| 观察值范围        | (0,255)                                |


经典的双人竞技计时游戏。

让球越过对手。

得分会给你 +1 奖励，给对手 -1 惩罚。

发球是有时间限制的：如果玩家在收到球后 2 秒内没有发球，他们会受到 -1 分的惩罚，计时器重置。这可以防止一个玩家无限期地拖延游戏，但也意味着这不再是一个纯零和游戏。

[官方视频奥运会手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

乒乓特有的参数如下：

``` python
pong_v3.env(num_players=2)
```

`num_players`：玩家数量（必须是 2 或 4）

### 动作空间（最小）

在任何给定回合中，智能体可以从 6 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 开火    |
| 2         | 向右移动 |
| 3         | 向左移动 |
| 4         | 向右开火 |
| 5         | 向左开火 |

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

avaliable_2p_versions = {
    "classic": 4,
    "two_paddles": 10,
    "soccer": 14,
    "foozpong": 19,
    "hockey": 27,
    "handball": 35,
    "volleyball": 39,
    "basketball": 45,
}
avaliable_4p_versions = {
    "classic": 6,
    "two_paddles": 11,
    "soccer": 16,
    "foozpong": 21,
    "hockey": 29,
    "quadrapong": 33,
    "handball": 37,
    "volleyball": 41,
    "basketball": 49,
}


def raw_env(num_players=2, game_version="classic", **kwargs):
    assert num_players == 2 or num_players == 4, "pong only supports 2 or 4 players"
    versions = avaliable_2p_versions if num_players == 2 else avaliable_4p_versions
    assert (
        game_version in versions
    ), f"pong version {game_version} not supported for number of players {num_players}. Available options are {list(versions)}"
    mode = versions[game_version]
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="pong",
        num_players=num_players,
        mode_num=mode,
        env_name=name,
        **kwargs,
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
