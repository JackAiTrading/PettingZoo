"""
太空入侵者环境。

这个环境模拟了一个经典的太空防御游戏，玩家控制防御炮台抵御
外星入侵者的进攻，需要通过精准的射击和战术配合来保护地球。

主要特点：
1. 多人合作
2. 太空防御
3. 战术配合
4. 资源管理

环境规则：
1. 基本设置
   - 防御炮台
   - 外星入侵者
   - 防护掩体
   - 特殊奖励

2. 交互规则
   - 炮台移动
   - 武器发射
   - 碰撞检测
   - 得分计算

3. 智能体行为
   - 目标选择
   - 火力分配
   - 防御策略
   - 资源利用

4. 终止条件
   - 消灭入侵者
   - 基地被毁
   - 时间耗尽
   - 玩家阵亡

环境参数：
- 观察空间：游戏画面状态
- 动作空间：炮台移动和射击控制
- 奖励：击毁敌人和存活奖励
- 最大步数：由关卡设置决定

环境特色：
1. 战斗系统
   - 武器类型
   - 弹道轨迹
   - 防护系统
   - 特殊能力

2. 敌人系统
   - 入侵者类型
   - 移动模式
   - 攻击方式
   - 难度递增

3. 战术元素
   - 火力分配
   - 防御布局
   - 资源管理
   - 优先级判断

4. 评估系统
   - 击杀数量
   - 存活时间
   - 防御效果
   - 整体表现

注意事项：
- 目标选择
- 火力分配
- 资源管理
- 防御策略
"""

# noqa: D212, D415
"""
太空入侵者游戏环境。

这个环境实现了经典雅达利游戏《太空入侵者》，玩家控制太空船防御来自外星人的入侵，
通过射击消灭不断逼近的外星舰队来保护地球。

主要特点：
1. 经典射击
2. 渐进难度
3. 分数系统
4. 策略防御

游戏规则：
1. 基本设置
   - 玩家太空船
   - 外星舰队
   - 防护掩体
   - 特殊奖励

2. 战斗规则
   - 射击消灭
   - 避免被击中
   - 保护掩体
   - 积分奖励

3. 角色控制
   - 左右移动
   - 发射激光
   - 躲避攻击
   - 使用掩护

4. 终止条件
   - 生命耗尽
   - 地球被侵略
   - 通关胜利
   - 主动退出

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：6个离散动作
- 奖励：击毁敌人和生存时间
- 最大步数：由关卡设计决定

游戏特色：
1. 敌人特性
   - 阵型移动
   - 加速下降
   - 随机射击
   - 特殊能力

2. 防御系统
   - 激光武器
   - 防护掩体
   - 能量护盾
   - 特殊武器

3. 关卡设计
   - 难度递增
   - 敌人变化
   - 奖励关卡
   - 最终BOSS

4. 得分机制
   - 击毁奖励
   - 连击加分
   - 生存奖励
   - 通关奖励

注意事项：
- 弹药管理
- 掩护利用
- 优先目标
- 躲避时机
"""

"""
# 太空侵略者（Space Invaders）

```{figure} atari_space_invaders.gif
:width: 140px
:name: space_invaders
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import space_invaders_v2` |
|----------------------|--------------------------------------------------|
| 动作类型           | 离散                                             |
| 并行 API          | 支持                                              |
| 手动控制          | 不支持                                           |
| 智能体            | `agents= ['first_0', 'second_0']`                |
| 智能体数量        | 2                                                |
| 动作形状          | (1,)                                             |
| 动作值范围        | [0,5]                                            |
| 观察形状          | (210, 160, 3)                                    |
| 观察值范围        | (0,255)                                          |


经典的 Atari 游戏，但有两艘由两个玩家控制的飞船，每个玩家都试图最大化他们的得分。

这个游戏具有合作性，玩家可以通过合作来通关以最大化他们的得分。普通外星人根据它们的起始高度可以得到 5-30 分，而在屏幕顶部飞过的飞船值 100 分。

然而，游戏也有竞争性的一面，当另一个玩家被外星人击中时，玩家会获得 200 分的奖励。所以破坏另一个玩家也是一种可能的策略。

飞船之间共享生命数，即当一艘飞船被击中 3 次时游戏结束。

[官方太空侵略者手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=460)

#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

太空侵略者特有的参数如下：

``` python
space_invaders_v2.env(alternating_control=False, moving_shields=True,
zigzaging_bombs=False, fast_bomb=False, invisible_invaders=False)
```

`alternating_control`：每次只有两个玩家中的一个有开火选项。如果你开火，你的对手就可以开火。但是，你不能永远保持开火能力，最终控制权会转移给你的对手。

`moving_shields`：护盾来回移动，提供的保护不太可靠。

`zigzaging_bombs`：入侵者的炸弹来回移动，更难避开。

`fast_bomb`：炸弹速度更快，更难避开。

`invisible_invaders`：入侵者是隐形的，更难击中。

### 动作空间（最小）

在任何给定回合中，智能体可以从 6 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 开火    |
| 2         | 向上移动 |
| 3         | 向右移动 |
| 4         | 向左移动 |
| 5         | 向下移动 |

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


def raw_env(
    alternating_control=False,
    moving_shields=True,
    zigzaging_bombs=False,
    fast_bomb=False,
    invisible_invaders=False,
    **kwargs,
):
    mode = 33 + (
        moving_shields * 1
        + zigzaging_bombs * 2
        + fast_bomb * 4
        + invisible_invaders * 8
        + alternating_control * 16
    )
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="space_invaders", num_players=2, mode_num=mode, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
