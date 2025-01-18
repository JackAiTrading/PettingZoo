"""
战斗坦克环境。

这个环境模拟了一个经典的双人坦克对战游戏，玩家控制坦克在战场上
进行对抗，需要利用地形优势和战术配合来击败对手。

主要特点：
1. 双人对战
2. 坦克作战
3. 地形利用
4. 战术对抗

环境规则：
1. 基本设置
   - 坦克单位
   - 武器系统
   - 战场地形
   - 障碍物

2. 交互规则
   - 移动控制
   - 炮塔旋转
   - 开火射击
   - 碰撞处理

3. 智能体行为
   - 地形探索
   - 瞄准射击
   - 战术掩护
   - 机动规避

4. 终止条件
   - 击毁对手
   - 时间耗尽
   - 坦克损毁
   - 轮次结束

环境参数：
- 观察空间：游戏画面状态
- 动作空间：移动和射击控制
- 奖励：击中得分和存活奖励
- 最大步数：由比赛设置决定

环境特色：
1. 战斗系统
   - 炮弹类型
   - 弹道轨迹
   - 装甲防护
   - 伤害模型

2. 移动机制
   - 速度控制
   - 转向系统
   - 地形影响
   - 惯性效应

3. 战术元素
   - 地形掩护
   - 火力压制
   - 战术机动
   - 位置控制

4. 评估系统
   - 命中精度
   - 存活时间
   - 战术运用
   - 整体表现

注意事项：
- 地形利用
- 弹道预判
- 战术配合
- 位置选择
"""
# noqa: D212, D415
"""
坦克战游戏环境。

这个环境实现了雅达利经典游戏《坦克大战》，两名玩家操控坦克在战场上进行对战。
游戏目标是击中对手的坦克，同时避免被对手击中。

主要特点：
1. 双人对战
2. 多种坦克类型
3. 不同战场地图
4. 弹道物理系统

游戏规则：
1. 基本设置
   - 两名玩家
   - 选择坦克类型
   - 选择战场地图
   - 有限的弹药量

2. 战斗规则
   - 击中对手得分
   - 被击中失分
   - 碰撞造成伤害
   - 弹药可补充

3. 坦克控制
   - 移动（前后）
   - 转向（左右）
   - 开火
   - 特殊技能

4. 终止条件
   - 达到分数上限
   - 时间耗尽
   - 一方被击毁
   - 双方同意终止

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：8个离散动作
- 奖励：击中得分，被击中失分
- 最大步数：由游戏时间决定

游戏特色：
1. 坦克类型
   - 轻型坦克（速度快）
   - 重型坦克（装甲厚）
   - 突击炮（火力强）
   - 反坦克炮（精准度高）

2. 战场地图
   - 开阔地带
   - 城市街道
   - 丛林地形
   - 沙漠地带

3. 地形特性
   - 障碍物可破坏
   - 掩体提供保护
   - 地形影响移动
   - 视线可被阻挡

注意事项：
- 战术位置重要
- 弹药管理关键
- 地形利用技巧
- 支持多种模式
"""
"""
# 坦克大战（Combat: Tank）

```{figure} atari_combat_tank.gif
:width: 140px
:name: combat_tank
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import combat_tank_v3`     |
|--------------------|---------------------------------------------------|
| 动作类型           | 离散                                              |
| 并行 API          | 支持                                               |
| 手动控制          | 不支持                                             |
| 智能体            | `agents= ['first_0', 'second_0']`                 |
| 智能体数量        | 2                                                 |
| 动作形状          | (1,)                                              |
| 动作值范围        | [0,5]                                             |
| 观察形状          | (210, 160, 3)                                     |
| 观察值范围        | (0,255)                                           |


*坦克大战*的经典坦克模式是一个对抗性游戏，预判和位置选择是关键。

玩家在地图上移动。当你的子弹击中对手时，
你得一分。注意，当对手被击中时，会被炸飞穿过障碍物，这可能会让对手处于有利位置反击你。

每当你得分时，你获得 +1 奖励，你的对手受到 -1 惩罚。

[官方坦克大战手册](https://atariage.com/manual_html_page.php?SoftwareID=935)


#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

combat-tank 特有的参数如下：

``` python
combat_tank_v2.env(has_maze=True, is_invisible=False, billiard_hit=True)
```

`has_maze`：设置为 true 时，地图将是迷宫而不是开放场地

`is_invisible`：如果为 true，坦克在开火或撞墙时才可见。

`billiard_hit`：如果为 true，子弹会从墙壁反弹，实际上，就像台球一样，只有在子弹从墙壁反弹后击中对手的坦克才算得分。

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
import warnings
from glob import glob

from pettingzoo.atari.base_atari_env import (
    BaseAtariEnv,
    base_env_wrapper_fn,
    parallel_wrapper_fn,
)


def raw_env(has_maze=True, is_invisible=False, billiard_hit=True, **kwargs):
    if has_maze is False and is_invisible is False and billiard_hit is False:
        warnings.warn(
            "坦克大战有一些有趣的参数可以考虑覆盖，包括 is_invisible（隐身）、billiard_hit（台球式击打）和 has_maze（迷宫）"
        )
    start_mapping = {
        (False, False): 1,
        (False, True): 8,
        (True, False): 10,
        (True, True): 13,
    }
    mode = start_mapping[(is_invisible, billiard_hit)] + has_maze
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
