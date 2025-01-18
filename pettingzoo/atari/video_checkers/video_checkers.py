# noqa: D212, D415
"""
电子跳棋环境。

这个环境模拟了一个经典的跳棋游戏，玩家在棋盘上轮流移动棋子，
需要通过战略布局和战术吃子来获得胜利。

主要特点：
1. 双人对抗
2. 战略博弈
3. 棋子控制
4. 战术运用

环境规则：
1. 基本设置
   - 棋盘大小
   - 棋子类型
   - 移动规则
   - 吃子规则

2. 交互规则
   - 棋子选择
   - 移动控制
   - 吃子判定
   - 胜负计算

3. 智能体行为
   - 局势判断
   - 棋子布局
   - 战术选择
   - 战略执行

4. 终止条件
   - 棋子耗尽
   - 无子可走
   - 认输投降
   - 胜负判定

环境参数：
- 观察空间：棋盘状态
- 动作空间：棋子移动选择
- 奖励：吃子和胜负奖励
- 最大步数：由规则设置决定

环境特色：
1. 棋盘系统
   - 位置关系
   - 移动规则
   - 吃子机制
   - 升王规则

2. 控制机制
   - 棋子选择
   - 移动确认
   - 吃子执行
   - 悔棋功能

3. 战术元素
   - 空间控制
   - 子力优势
   - 战术配合
   - 战略布局

4. 评估系统
   - 子力对比
   - 局势评估
   - 战术效果
   - 整体表现

注意事项：
- 局势判断
- 战术选择
- 子力保护
- 战略执行
"""
"""
视频跳棋游戏环境。

这个环境实现了雅达利游戏《视频跳棋》，两名玩家在棋盘上进行经典的跳棋对战，
通过战术性的移动和吃子来获得胜利。

主要特点：
1. 双人对战
2. 经典规则
3. 策略思考
4. 多步规划

游戏规则：
1. 基本设置
   - 8x8棋盘
   - 双方棋子
   - 王棋升级
   - 计分系统

2. 移动规则
   - 斜向移动
   - 跳吃对方
   - 连续吃子
   - 王棋特权

3. 棋子控制
   - 选择棋子
   - 确定方向
   - 执行移动
   - 完成回合

4. 终止条件
   - 无子可走
   - 无法移动
   - 认输投降
   - 和棋协议

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：8个离散动作
- 奖励：吃子得分和获胜
- 最大步数：由对局进程决定

游戏特色：
1. 棋盘布局
   - 黑白相间
   - 对角布置
   - 活动范围
   - 边界限制

2. 棋子特性
   - 普通棋子
   - 王棋能力
   - 移动范围
   - 吃子规则

3. 战术元素
   - 阵型布局
   - 子力价值
   - 进攻策略
   - 防守部署

4. 胜利条件
   - 吃光对方
   - 限制移动
   - 战术降服
   - 局势优势

注意事项：
- 局势判断
- 战术选择
- 子力平衡
- 关键时机
"""
"""
# Video Checkers

```{figure} atari_video_checkers.gif
:width: 140px
:name: video_checkers
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import video_checkers_v4` |
|----------------------|--------------------------------------------------|
| Actions              | Discrete                                         |
| Parallel API         | Yes                                              |
| Manual Control       | No                                               |
| Agents               | `agents= ['first_0', 'second_0']`                |
| Agents               | 2                                                |
| Action Shape         | (1,)                                             |
| Action Values        | [0,4]                                            |
| Observation Shape    | (210, 160, 3)                                    |
| Observation Values   | (0,255)                                          |


A classical strategy game with arcade style controls.

Capture all of your opponents pieces by jumping over them. To move a piece, you must select a piece by hovering the cursor and pressing fire (action 1), moving the cursor, and pressing fire again. Note that the buttons must be held for multiple frames to be registered.

If you win by capturing all your opponent's pieces, you are rewarded +1 and your opponent -1.

This is a timed game: if a player does not take a turn after 10 seconds, then that player is rewarded -1 points, their opponent is rewarded nothing, and the timer resets. This prevents one player from indefinitely stalling the game, but also means it is no longer a purely zero sum game.


[Official video checkers manual](https://atariage.com/manual_html_page.php?SoftwareID=1427)

#### Environment parameters

Environment parameters are common to all Atari environments and are described in the [base Atari documentation](../atari) .

### Action Space (Minimal)

In any given turn, an agent can choose from one of 5 actions.

| Action    | Behavior  |
|:---------:|-----------|
| 0         | Fire |
| 1         | Move up |
| 2         | Move right |
| 3         | Move left |
| 4         | Move down |


### Version History

* v4: Minimal Action Space (1.18.0)
* v3: No action timer (1.9.0)
* v2: Fixed checkers rewards (1.5.0)
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
        game="video_checkers", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
