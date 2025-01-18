"""
篮球乒乓环境。

这个环境模拟了一个创新的篮球乒乓混合游戏，玩家控制球拍在篮球场地中
进行对抗，需要通过灵活的移动和精准的击球来投篮得分。

主要特点：
1. 双人对战
2. 篮球元素
3. 物理碰撞
4. 投篮技巧

环境规则：
1. 基本设置
   - 球拍单位
   - 篮球
   - 篮筐
   - 场地边界

2. 交互规则
   - 球拍移动
   - 球体反弹
   - 投篮判定
   - 得分计算

3. 智能体行为
   - 位置调整
   - 球速控制
   - 投篮角度
   - 进攻策略

4. 终止条件
   - 达到分数
   - 时间耗尽
   - 回合结束
   - 比赛完成

环境参数：
- 观察空间：游戏画面状态
- 动作空间：球拍移动控制
- 奖励：投篮得分和配合奖励
- 最大步数：由比赛设置决定

环境特色：
1. 物理系统
   - 球体运动
   - 投篮轨迹
   - 反弹效果
   - 碰撞检测

2. 控制机制
   - 球拍移动
   - 击球力度
   - 投篮角度
   - 位置把控

3. 战术元素
   - 场地利用
   - 投篮时机
   - 进攻配合
   - 防守策略

4. 评估系统
   - 投篮命中率
   - 助攻次数
   - 技术运用
   - 整体表现

注意事项：
- 投篮角度
- 力度控制
- 位置选择
- 战术配合
"""
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
