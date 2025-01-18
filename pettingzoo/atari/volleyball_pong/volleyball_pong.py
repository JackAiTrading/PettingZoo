"""
排球乒乓环境。

这个环境模拟了一个创新的排球乒乓混合游戏，玩家控制球拍在排球场地中
进行对抗，需要通过灵活的移动和精准的击球来得分。

主要特点：
1. 双人对战
2. 排球元素
3. 物理碰撞
4. 技术对抗

环境规则：
1. 基本设置
   - 球拍单位
   - 排球
   - 网线
   - 场地边界

2. 交互规则
   - 球拍移动
   - 球体反弹
   - 过网判定
   - 得分计算

3. 智能体行为
   - 位置调整
   - 球速控制
   - 击球角度
   - 进攻策略

4. 终止条件
   - 达到分数
   - 时间耗尽
   - 回合结束
   - 比赛完成

环境参数：
- 观察空间：游戏画面状态
- 动作空间：球拍移动控制
- 奖励：得分和防守奖励
- 最大步数：由比赛设置决定

环境特色：
1. 物理系统
   - 球体运动
   - 网球反弹
   - 重力效果
   - 碰撞检测

2. 控制机制
   - 球拍移动
   - 击球力度
   - 击球角度
   - 位置把控

3. 战术元素
   - 场地利用
   - 击球时机
   - 进攻变化
   - 防守策略

4. 评估系统
   - 得分情况
   - 防守成功率
   - 技术运用
   - 整体表现

注意事项：
- 网前击球
- 力度控制
- 位置选择
- 战术配合
"""
"""
排球乒乓游戏环境。

这个环境实现了雅达利游戏《排球乒乓》，结合了乒乓球和排球的玩法，
两名玩家用球拍击球，通过网将球打到对方场地。

主要特点：
1. 双人对战
2. 排球元素
3. 物理引擎
4. 网前对抗

游戏规则：
1. 基本设置
   - 两名玩家
   - 每人一个球拍
   - 一个排球
   - 中间有网

2. 得分规则
   - 球落地得分
   - 球出界失分
   - 连续得分奖励
   - 发球失误扣分

3. 球拍控制
   - 上下移动
   - 击球角度
   - 力度控制
   - 发球技巧

4. 终止条件
   - 达到目标分数
   - 时间耗尽
   - 一方认输
   - 双方同意终止

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：3个离散动作（上、下、不动）
- 奖励：得分为1，失分为-1
- 最大步数：由分数上限决定

游戏特色：
1. 球的物理特性
   - 抛物线轨迹
   - 重力效果
   - 网前反弹
   - 旋转效果

2. 场地设计
   - 中间球网
   - 界内区域
   - 发球区
   - 得分区

3. 战术元素
   - 发球战术
   - 扣球技巧
   - 防守站位
   - 快速反应

注意事项：
- 发球很重要
- 网前控制关键
- 需要预判落点
- 支持练习模式
"""
"""
# Volleyball Pong

```{figure} atari_volleyball_pong.gif
:width: 140px
:name: volleyball_pong
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import volleyball_pong_v2`        |
|----------------------|----------------------------------------------------------|
| Actions              | Discrete                                                 |
| Parallel API         | Yes                                                      |
| Manual Control       | No                                                       |
| Agents               | `agents= ['first_0', 'second_0', 'third_0', 'fourth_0']` |
| Agents               | 4                                                        |
| Action Shape         | (1,)                                                     |
| Action Values        | [0,5]                                                    |
| Observation Shape    | (210, 160, 3)                                            |
| Observation Values   | (0,255)                                                  |


Four player team battle.

Get the ball onto your opponent's floor to score. In addition to being able to move left and right, each player can also jump higher to affect the ball's motion above the net.
This is a team game, so a given team must try to coordinate to get the ball away from their scoring areas towards their opponent's.
Specifically `first_0` and `third_0` are on one team and `second_0` and `fourth_0` are on the other.

Scoring a point gives your team +1 reward and your opponent team -1 reward.

Serves are timed: If the player does not serve within 2 seconds of receiving the ball, their team receives -1 points, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero sum game.


[Official Video Olympics manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

#### Environment parameters

Some environment parameters are common to all Atari environments and are described in the [base Atari documentation](../atari).

Parameters specific to Volleyball Pong are

``` python
volleyball_pong_v3.env(num_players=4)
```

`num_players`:  Number of players (must be either 2 or 4)

### Action Space (Minimal)

In any given turn, an agent can choose from one of 6 actions.

| Action    | Behavior  |
|:---------:|-----------|
| 0         | No operation |
| 1         | Fire |
| 2         | Move up |
| 3         | Move right |
| 4         | Move left |
| 5         | Move down |

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


def raw_env(num_players=4, **kwargs):
    assert num_players == 2 or num_players == 4, "pong only supports 2 or 4 players"
    mode_mapping = {2: 39, 4: 41}
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
