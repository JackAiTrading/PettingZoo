"""
骑士决斗环境。

这个环境模拟了一个经典的骑士对决游戏，玩家骑乘飞行坐骑在竞技场中
进行对抗，需要通过灵活的机动和精准的攻击来击败对手。

主要特点：
1. 双人对战
2. 飞行战斗
3. 物理引擎
4. 技术对抗

环境规则：
1. 基本设置
   - 骑士角色
   - 飞行坐骑
   - 竞技场
   - 平台障碍

2. 交互规则
   - 飞行控制
   - 攻击判定
   - 碰撞检测
   - 得分计算

3. 智能体行为
   - 高度控制
   - 攻击时机
   - 躲避策略
   - 位置调整

4. 终止条件
   - 击败对手
   - 时间耗尽
   - 回合结束
   - 比赛完成

环境参数：
- 观察空间：游戏画面状态
- 动作空间：飞行控制和攻击
- 奖励：击败对手和存活奖励
- 最大步数：由比赛设置决定

环境特色：
1. 战斗系统
   - 攻击方式
   - 飞行轨迹
   - 碰撞效果
   - 伤害计算

2. 控制机制
   - 飞行操控
   - 高度调节
   - 速度控制
   - 方向转换

3. 战术元素
   - 地形利用
   - 攻击时机
   - 躲避技巧
   - 空间控制

4. 评估系统
   - 击败数量
   - 存活时间
   - 技术运用
   - 整体表现

注意事项：
- 飞行控制
- 攻击时机
- 躲避策略
- 位置把控
"""
"""
骑士对决游戏环境。

这个环境实现了雅达利游戏《骑士对决》，玩家骑乘飞行坐骑进行空中对决，
通过高超的飞行技巧和战术来击败对手。

主要特点：
1. 双人对战
2. 物理引擎
3. 飞行控制
4. 战术对抗

游戏规则：
1. 基本设置
   - 两名骑士
   - 飞行坐骑
   - 竞技场地
   - 得分系统

2. 战斗规则
   - 高度优势
   - 碰撞判定
   - 击落得分
   - 复活机制

3. 角色控制
   - 飞行方向
   - 速度控制
   - 攻击动作
   - 防御姿态

4. 终止条件
   - 生命耗尽
   - 时间结束
   - 达到分数
   - 一方认输

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：6个离散动作
- 奖励：击败对手和生存时间
- 最大步数：由比赛时间决定

游戏特色：
1. 飞行系统
   - 重力效果
   - 惯性运动
   - 空气阻力
   - 高度控制

2. 战斗机制
   - 高度优势
   - 撞击判定
   - 击退效果
   - 连击奖励

3. 战术元素
   - 位置控制
   - 时机把握
   - 进攻策略
   - 防守反击

4. 场地特性
   - 平台布局
   - 危险区域
   - 复活点
   - 能量补给

注意事项：
- 高度控制
- 速度管理
- 战术选择
- 风险评估
"""
"""
# Joust

```{figure} atari_joust.gif
:width: 140px
:name: joust
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import joust_v3` |
|----------------------|-----------------------------------------|
| Actions              | Discrete                                |
| Parallel API         | Yes                                     |
| Manual Control       | No                                      |
| Agents               | `agents= ['first_0', 'second_0']`       |
| Agents               | 2                                       |
| Action Shape         | (1,)                                    |
| Action Values        | [0,17]                                  |
| Observation Shape    | (210, 160, 3)                           |
| Observation Values   | (0,255)                                 |


Mixed sum game involving scoring points in an unforgiving world. Careful positioning, timing,
and control is essential, as well as awareness of your opponent.

In Joust, you score points by hitting the opponent and NPCs when
you are above them. If you are below them, you lose a life.
In a game, there are a variety of waves with different enemies
and different point scoring systems. However, expect that you can earn
around 3000 points per wave.

[Official joust manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=253)

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
* v2: Breaking changes to entire API (1.4.0)
* v1: Fixes to how all environments handle premature death (1.3.0)
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
        game="joust", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
