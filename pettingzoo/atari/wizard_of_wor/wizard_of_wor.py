"""
巫师之战环境。

这个环境模拟了一个经典的地下城探索游戏，玩家扮演巫师在迷宫中
寻找敌人并进行魔法对决，需要通过战术配合和魔法运用来击败对手。

主要特点：
1. 多人对战
2. 迷宫探索
3. 魔法战斗
4. 团队合作

环境规则：
1. 基本设置
   - 巫师角色
   - 迷宫地图
   - 魔法系统
   - 怪物单位

2. 交互规则
   - 移动控制
   - 魔法施放
   - 视野探索
   - 战斗判定

3. 智能体行为
   - 地图探索
   - 魔法选择
   - 战术配合
   - 敌人追踪

4. 终止条件
   - 消灭敌人
   - 时间耗尽
   - 巫师阵亡
   - 关卡完成

环境参数：
- 观察空间：游戏画面状态
- 动作空间：移动和魔法控制
- 奖励：击杀得分和探索奖励
- 最大步数：由关卡设置决定

环境特色：
1. 战斗系统
   - 魔法类型
   - 伤害计算
   - 效果判定
   - 连击机制

2. 探索机制
   - 迷宫生成
   - 视野范围
   - 地图记忆
   - 路径规划

3. 战术元素
   - 地形利用
   - 魔法配合
   - 团队协作
   - 资源管理

4. 评估系统
   - 击杀数量
   - 探索完成度
   - 生存时间
   - 团队贡献

注意事项：
- 地图熟悉
- 魔法运用
- 团队配合
- 战术规划
"""
"""
巫师之战（Wizard of Wor）

```{figure} atari_wizard_of_wor.gif
:width: 140px
:name: wizard_of_wor
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | from pettingzoo.atari import wizard_of_wor_v3 |
|----------------------|-----------------------------------------------|
| 动作类型           | 离散                                          |
| 并行 API          | 支持                                           |
| 手动控制          | 不支持                                         |
| 智能体            | agents= ['first_0', 'second_0']               |
| 智能体数量        | 2                                             |
| 动作形状          | (1,)                                          |
| 动作值范围        | [0,8]                                         |
| 观察形状          | (210, 160, 3)                                 |
| 观察值范围        | (0,255)                                       |


既要与 NPC 战斗，又要与其他玩家战斗。精确的时机
和控制至关重要，同时还要注意你的对手。

你可以通过用子弹击中对手和 NPC 来得分。击中 NPC 可以得到 200 到 2500 分不等，具体取决于 NPC 的类型，击中玩家可以得到 1000 分。

如果你被子弹击中，你就会失去一条生命。当两个玩家都失去 3 条生命时，游戏结束。

请注意，除了通过攻击其他玩家获益的竞争性方面外，游戏还有一个合作性方面，即通关意味着两个玩家都将有更多机会得分。

[官方巫师之战手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=593)

#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

### 动作空间

在任何给定回合中，智能体可以从 9 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 开火    |
| 1         | 向上移动 |
| 2         | 向右移动 |
| 3         | 向左移动 |
| 4         | 向下移动 |
| 5         | 向右上移动 |
| 6         | 向左上移动 |
| 7         | 向右下移动 |
| 8         | 向左下移动 |

### 版本历史

* v3：最小动作空间 (1.18.0)
* v2：对整个 API 进行重大更改 (1.4.0)
* v1：修复了所有环境处理过早死亡的方式 (1.3.0)
* v0：初始版本发布 (1.0.0)


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
        game="wizard_of_wor", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
