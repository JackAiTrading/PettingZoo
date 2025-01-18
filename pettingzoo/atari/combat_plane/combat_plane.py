"""
战斗机环境。

这个环境模拟了一个经典的双人空战游戏，玩家控制战斗机在竞技场中
进行对抗，需要通过灵活的操作和精准的射击来击败对手。

主要特点：
1. 双人对战
2. 空战竞技
3. 弹道物理
4. 地图策略

环境规则：
1. 基本设置
   - 战斗机
   - 武器系统
   - 竞技场地
   - 障碍物

2. 交互规则
   - 飞行控制
   - 武器发射
   - 碰撞检测
   - 伤害计算

3. 智能体行为
   - 机动飞行
   - 瞄准射击
   - 躲避攻击
   - 战术运用

4. 终止条件
   - 击落对手
   - 时间耗尽
   - 飞机损毁
   - 轮次结束

环境参数：
- 观察空间：游戏画面状态
- 动作空间：飞行和射击控制
- 奖励：击中得分和存活奖励
- 最大步数：由比赛设置决定

环境特色：
1. 战斗系统
   - 武器类型
   - 弹道轨迹
   - 伤害模型
   - 命中判定

2. 飞行机制
   - 速度控制
   - 转向系统
   - 加速减速
   - 惯性效应

3. 战术元素
   - 地形利用
   - 火力压制
   - 机动规避
   - 位置控制

4. 评估系统
   - 命中率
   - 存活时间
   - 战术运用
   - 整体表现

注意事项：
- 速度控制
- 弹道预判
- 地形利用
- 战术选择
"""
# noqa: D212, D415
"""
空战游戏环境。

这个环境实现了雅达利经典游戏《空战》的飞机对战模式，两名玩家操控飞机进行空中战斗。
游戏目标是击落对手的飞机，同时避免被对手击中。

主要特点：
1. 双人对战
2. 多种飞机类型
3. 不同战场地图
4. 物理碰撞系统

游戏规则：
1. 基本设置
   - 两名玩家
   - 选择飞机类型
   - 选择战场地图
   - 有限的弹药量

2. 战斗规则
   - 击中对手得分
   - 被击中失分
   - 碰撞双方损失
   - 弹药可补充

3. 飞机控制
   - 转向（左右）
   - 加速/减速
   - 开火
   - 特殊动作

4. 终止条件
   - 达到分数上限
   - 时间耗尽
   - 一方被击落
   - 双方同意终止

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：8个离散动作
- 奖励：击中得分，被击中失分
- 最大步数：由游戏时间决定

游戏特色：
1. 飞机类型
   - 战斗机（快速灵活）
   - 轰炸机（火力强大）
   - 侦察机（隐身能力）
   - 运输机（装甲厚重）

2. 战场地图
   - 开阔平原
   - 城市建筑
   - 山地峡谷
   - 海岛礁石

注意事项：
- 需要掌握飞行技巧
- 弹药管理重要
- 地形可用于掩护
- 支持多种游戏模式
"""
"""
# 空战：飞机（Combat: Plane）

```{figure} atari_combat_plane.gif
:width: 140px
:name: combat_plane
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import combat_jet_v1` |
|--------------------|----------------------------------------------|
| 动作类型           | 离散                                         |
| 并行 API          | 支持                                          |
| 手动控制          | 不支持                                        |
| 智能体            | `agents= ['first_0', 'second_0']`            |
| 智能体数量        | 2                                            |
| 动作形状          | (1,)                                         |
| 动作值范围        | [0,17]                                       |
| 观察形状          | (256, 160, 3)                                |
| 观察值范围        | (0,255)                                      |


*空战*的飞机模式是一个对抗性游戏，时机掌握、
位置选择和追踪对手的复杂
动作是关键。

玩家在地图上飞行，可以控制飞行方向
但不能控制速度。

当你的子弹击中对手时，
你得一分。

每当你得分时，你获得 +1 奖励，你的对手受到 -1 惩罚。

[官方空战游戏手册](https://atariage.com/manual_html_page.php?SoftwareID=935)


#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

combat-plane 特有的参数如下：

``` python
combat_plane_v2.env(game_version="jet", guided_missile=True)
```

`game_version`：接受的参数为 "jet" 或 "bi-plane"。决定飞机是双翼机还是喷气式飞机。（喷气式飞机移动更快）

`guided_missile`：导弹发射后是否可以被引导，或者是否沿固定路径飞行。

### 动作空间

在任何给定回合中，智能体可以从 18 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 开火    |
| 2         | 向上移动 |
| 3         | 向右移动 |
| 4         | 向左移动 |
| 5         | 向下移动 |
| 6         | 向右上移动 |
| 7         | 向左上移动 |
| 8         | 向右下移动 |
| 9         | 向左下移动 |
| 10        | 向上开火 |
| 11        | 向右开火 |
| 12        | 向左开火 |
| 13        | 向下开火 |
| 14        | 向右上开火 |
| 15        | 向左上开火 |
| 16        | 向右下开火 |
| 17        | 向左下开火 |

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

avaliable_versions = {
    "bi-plane": 15,
    "jet": 21,
}


def raw_env(game_version="bi-plane", guided_missile=True, **kwargs):
    assert (
        game_version in avaliable_versions
    ), "game_version 必须是 'jet' 或 'bi-plane'"
    mode = avaliable_versions[game_version] + (0 if guided_missile else 1)
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="combat", num_players=2, mode_num=mode, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
