"""
网球环境。

这个环境模拟了一个经典的网球对抗游戏，玩家控制网球运动员
在球场上进行对抗，需要通过灵活的移动和精准的击球来得分。

主要特点：
1. 双人对战
2. 网球运动
3. 物理引擎
4. 技术对抗

环境规则：
1. 基本设置
   - 球员角色
   - 网球
   - 球网
   - 场地边界

2. 交互规则
   - 球员移动
   - 击球控制
   - 发球规则
   - 得分计算

3. 智能体行为
   - 位置调整
   - 击球选择
   - 发球策略
   - 防守反应

4. 终止条件
   - 达到分数
   - 时间耗尽
   - 回合结束
   - 比赛完成

环境参数：
- 观察空间：游戏画面状态
- 动作空间：球员移动和击球控制
- 奖励：得分和技术奖励
- 最大步数：由比赛设置决定

环境特色：
1. 网球系统
   - 击球类型
   - 球路轨迹
   - 旋转效果
   - 反弹物理

2. 控制机制
   - 球员移动
   - 击球力度
   - 击球角度
   - 时机把控

3. 战术元素
   - 场地利用
   - 发球变化
   - 进攻组织
   - 防守策略

4. 评估系统
   - 得分情况
   - 技术运用
   - 战术执行
   - 整体表现

注意事项：
- 位置选择
- 击球时机
- 技术运用
- 战术配合
"""
"""
网球游戏环境。

这个环境实现了雅达利游戏《网球》，两名玩家在网球场上进行对抗，
通过控制球员的移动和击球来赢得比赛。

主要特点：
1. 双人对战
2. 物理引擎
3. 球场规则
4. 技术对抗

游戏规则：
1. 基本设置
   - 两名球员
   - 网球场地
   - 发球规则
   - 计分系统

2. 比赛规则
   - 发球得分
   - 界内界外
   - 网前截击
   - 回球失误

3. 球员控制
   - 移动位置
   - 击球力度
   - 球拍角度
   - 跑位战术

4. 终止条件
   - 得分达标
   - 时间结束
   - 一方认输
   - 双方同意

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：8个离散动作
- 奖励：得分和回合胜利
- 最大步数：由比赛规则决定

游戏特色：
1. 球场系统
   - 界线判定
   - 发球区域
   - 网前区域
   - 底线区域

2. 击球特性
   - 球速变化
   - 旋转效果
   - 落点预测
   - 反弹角度

3. 战术元素
   - 发球战术
   - 跑位策略
   - 网前截击
   - 底线抽击

4. 比赛机制
   - 发球轮换
   - 分数计算
   - 局数设定
   - 胜负判定

注意事项：
- 发球重要
- 位置把控
- 技术运用
- 体力管理
"""
"""
# Tennis

```{figure} atari_tennis.gif
:width: 140px
:name: tennis
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import tennis_v3` |
|----------------------|------------------------------------------|
| Actions              | Discrete                                 |
| Parallel API         | Yes                                      |
| Manual Control       | No                                       |
| Agents               | `agents= ['first_0', 'second_0']`        |
| Agents               | 2                                        |
| Action Shape         | (1,)                                     |
| Action Values        | [0,17]                                   |
| Observation Shape    | (210, 160, 3)                            |
| Observation Values   | (0,255)                                  |


A competitive game of positioning and prediction.

Goal: Get the ball past your opponent. Don't let the ball get past you.

When a point is scored (by the ball exiting the area), you get +1 reward and your opponent gets -1 reward. Unlike normal tennis matches, the number of games won is not directly rewarded.

Serves are timed: If the player does not serve within 3 seconds of receiving the ball, they receive -1 points, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero sum game.

[Official tennis manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=555)

#### Environment parameters

Environment parameters are common to all Atari environments and are described in the [base Atari documentation](../atari) .

### Action Space

In any given turn, an agent can choose from one of 18 actions.

| Action    | Behavior  |
|:---------:|-----------|
| 0         | No operation |
| 1         | Fire |
| 2         | Move up |
| 3         | Move right |
| 4         | Move left |
| 5         | Move down |
| 6         | Move upright |
| 7         | Move upleft |
| 8         | Move downright |
| 9         | Move downleft |
| 10        | Fire up |
| 11        | Fire right |
| 12        | Fire left |
| 13        | Fire down |
| 14        | Fire upright |
| 15        | Fire upleft |
| 16        | Fire downright |
| 17        | Fire downleft |

### Version History

* v3: Minimal Action Space (1.18.0)
* v2: No action timer (1.9.0)
* v1: Breaking changes to entire API (1.4.0)
* v0: Initial versions release (1.0.0)


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
        game="tennis", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
