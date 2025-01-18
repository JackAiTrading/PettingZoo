"""
夺旗环境。

这个环境模拟了一个双人夺旗游戏，玩家需要在迷宫中寻找对方的旗帜，
同时保护自己的旗帜不被对手夺取。

主要特点：
1. 双人对抗
2. 迷宫探索
3. 战略布局
4. 战术执行

环境规则：
1. 基本设置
   - 玩家角色
   - 迷宫地图
   - 旗帜位置
   - 障碍物

2. 交互规则
   - 角色移动
   - 旗帜夺取
   - 碰撞检测
   - 胜负判定

3. 智能体行为
   - 路径规划
   - 搜索策略
   - 防守布局
   - 进攻时机

4. 终止条件
   - 旗帜被夺
   - 时间耗尽
   - 回合结束
   - 胜负判定

环境参数：
- 观察空间：游戏画面状态
- 动作空间：角色移动控制
- 奖励：夺旗和防守奖励
- 最大步数：由比赛设置决定

环境特色：
1. 迷宫系统
   - 地图生成
   - 路径验证
   - 视野范围
   - 碰撞检测

2. 控制机制
   - 角色移动
   - 旗帜交互
   - 障碍绕行
   - 视野控制

3. 战术元素
   - 地形利用
   - 搜索策略
   - 防守布局
   - 进攻路线

4. 评估系统
   - 夺旗速度
   - 防守效果
   - 探索效率
   - 整体表现

注意事项：
- 路径规划
- 搜索策略
- 防守布局
- 进攻时机
"""
# noqa: D212, D415
"""
夺旗游戏环境。

这个环境实现了雅达利游戏《夺旗》，两名玩家在迷宫中竞争寻找和夺取对方的旗帜，
通过策略性的移动和道具使用来完成任务。

主要特点：
1. 双人竞争
2. 迷宫探索
3. 道具系统
4. 策略对抗

游戏规则：
1. 基本设置
   - 两名玩家
   - 随机迷宫
   - 双方旗帜
   - 多种道具

2. 夺旗规则
   - 寻找旗帜
   - 抢夺旗帜
   - 保护旗帜
   - 返回基地

3. 角色控制
   - 上下左右移动
   - 使用道具
   - 攻击对手
   - 防守基地

4. 终止条件
   - 夺旗成功
   - 时间耗尽
   - 生命耗尽
   - 一方认输

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：4个离散动作
- 奖励：夺旗得分和生存时间
- 最大步数：由迷宫大小决定

游戏特色：
1. 迷宫设计
   - 随机生成
   - 多条路径
   - 隐藏区域
   - 战略位置

2. 道具系统
   - 速度提升
   - 视野扩大
   - 攻击道具
   - 防御道具

3. 战术元素
   - 路线选择
   - 道具使用
   - 追逐战术
   - 防守策略

4. 对抗机制
   - 直接冲突
   - 迷惑对手
   - 陷阱设置
   - 战术撤退

注意事项：
- 路线规划
- 道具管理
- 对手预判
- 时机把握
"""
"""
夺旗游戏环境。

这个模块实现了一个双人夺旗游戏，玩家需要在迷宫中寻找并夺取对方的旗帜。
游戏基于 Atari 2600 的 Flag Capture 游戏。

# Flag Capture Game

```{figure} atari_flag_capture.gif
:width: 140px
:name: flag_capture
```

这个环境是 Atari 环境的一部分。请先阅读 Atari 环境的文档。

| Import                 | `from pettingzoo.atari import flag_capture_v2` |
|----------------------|------------------------------------------------|
| Action Space          | Discrete                                         |
| Parallel API          | Yes                                             |
| Manual Control        | No                                              |
| Agents                | `agents= ['first_0', 'second_0']`                |
| Agents Number         | 2                                               |
| Action Shape          | (1,)                                            |
| Action Values         | [0,9]                                           |
| Observation Shape    | (210, 160, 3)                                   |
| Observation Values   | (0,255)                                         |

这个游戏需要记忆和信息。

一个旗帜被隐藏在地图上。
你可以在地图上移动并检查
方块。如果你找到旗帜，
你将获得一分（你的对手将失去一分）。
如果是炸弹，你将被送回起始位置。
否则，它将给你一个关于旗帜位置的提示，
可能是方向或距离。
你的玩家需要能够使用自己的搜索和对手的搜索来快速有效地缩小旗帜的位置。

[官方 Flag Capture 游戏手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=183)


#### Environment Parameters

环境参数是所有 Atari 环境共有的，并在 [Base Atari 文档](../atari) 中描述。

### Action Space (Minimal)

在任何给定的回合中，代理可以选择 10 个动作。

| Action     | Behavior    |
|:---------:|-----------|
| 0         | No operation |
| 1         | Fire        |
| 2         | Move up     |
| 3         | Move right  |
| 4         | Move left   |
| 5         | Move down   |
| 6         | Move up-right|
| 7         | Move up-left|
| 8         | Move down-right|
| 9         | Move down-left|

### Version History

* v2: 最小化动作空间 (1.18.0)
* v1: 主要 API 更改 (1.4.0)
* v0: 初始版本发布 (1.0.0)

"""

import os
from glob import glob

from pettingzoo.atari.base_atari_env import (
    BaseAtariEnv,
    base_env_wrapper_fn,
    parallel_wrapper_fn,
)


def raw_env(**kwargs):
    """夺旗游戏环境的原始实现。

    参数:
        **kwargs: 传递给父类的参数

    返回:
        BaseAtariEnv: 夺旗游戏环境
    """
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="flag_capture", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
