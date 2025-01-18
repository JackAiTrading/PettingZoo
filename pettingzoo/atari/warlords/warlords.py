# noqa: D212, D415
"""
诸侯战争环境。

这个环境模拟了一个多人城堡防御游戏，玩家需要保护自己的城堡
同时攻击其他玩家，通过战术防守和精准进攻来获得胜利。

主要特点：
1. 多人对抗
2. 城堡防御
3. 战术博弈
4. 资源管理

环境规则：
1. 基本设置
   - 城堡单位
   - 防御墙
   - 攻击球
   - 场地边界

2. 交互规则
   - 防御移动
   - 球体反弹
   - 碰撞检测
   - 得分计算

3. 智能体行为
   - 防御布局
   - 攻击策略
   - 资源利用
   - 战术调整

4. 终止条件
   - 城堡摧毁
   - 时间耗尽
   - 回合结束
   - 胜负判定

环境参数：
- 观察空间：游戏画面状态
- 动作空间：防御移动和攻击控制
- 奖励：城堡存活和击毁奖励
- 最大步数：由比赛设置决定

环境特色：
1. 战斗系统
   - 防御机制
   - 攻击轨迹
   - 反弹效果
   - 伤害计算

2. 控制机制
   - 防御移动
   - 攻击角度
   - 力度控制
   - 位置把控

3. 战术元素
   - 防御布局
   - 攻击时机
   - 资源管理
   - 战术配合

4. 评估系统
   - 城堡耐久
   - 击毁数量
   - 防御效果
   - 整体表现

注意事项：
- 防御布局
- 攻击时机
- 资源管理
- 战术选择
"""

"""
军阀游戏环境。

这个环境实现了雅达利游戏《军阀》，四名玩家在四个角落各自拥有一座城堡，
通过控制挡板来保护城堡并试图摧毁其他玩家的城堡。

主要特点：
1. 四人对战
2. 城堡防御
3. 物理引擎
4. 策略对抗

游戏规则：
1. 基本设置
   - 四座城堡
   - 移动挡板
   - 飞行火球
   - 防护墙壁

2. 战斗规则
   - 保护城堡
   - 反弹火球
   - 摧毁对手
   - 最后生存

3. 挡板控制
   - 围绕移动
   - 反弹角度
   - 速度控制
   - 防御策略

4. 终止条件
   - 城堡摧毁
   - 最后生存
   - 时间结束
   - 主动退出

环境参数：
- 观察空间：游戏画面（RGB数组）
- 动作空间：4个离散动作
- 奖励：城堡存活和摧毁对手
- 最大步数：由比赛时间决定

游戏特色：
1. 城堡系统
   - 防御能力
   - 损伤状态
   - 修复机制
   - 防护等级

2. 火球特性
   - 速度变化
   - 反弹效果
   - 破坏力度
   - 连锁反应

3. 战术元素
   - 位置控制
   - 火球引导
   - 联合作战
   - 防守反击

4. 场地特性
   - 角落城堡
   - 中央区域
   - 反弹墙壁
   - 战略位置

注意事项：
- 防御优先
- 攻击时机
- 合作可能
- 风险评估
"""

"""
# Warlords

```{figure} atari_warlords.gif
:width: 140px
:name: warlords
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.atari import warlords_v3`               |
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


Four player last man standing!

Defend your fortress from the ball and hit it towards your opponents.

When your fortress falls, you receive -1 reward and are done. If you are the last player standing, you receive +1 reward.

[Official wizard_of_wor manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=598)

#### Environment parameters

Environment parameters are common to all Atari environments and are described in the [base Atari documentation](../atari) .

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
        game="warlords", num_players=4, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
