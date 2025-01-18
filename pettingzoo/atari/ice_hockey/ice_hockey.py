"""
冰球环境。

这个环境模拟了一个经典的双人冰球对抗游戏，玩家控制冰球运动员在冰场上
进行对抗，需要通过传球配合和射门技巧来得分。

主要特点：
1. 双人对战
2. 冰球运动
3. 团队配合
4. 技术对抗

环境规则：
1. 基本设置
   - 球员角色
   - 冰球
   - 球门
   - 冰场边界

2. 交互规则
   - 球员移动
   - 传球控制
   - 射门动作
   - 碰撞判定

3. 智能体行为
   - 跑位策略
   - 传球选择
   - 射门时机
   - 防守配合

4. 终止条件
   - 达到分数
   - 时间耗尽
   - 回合结束
   - 比赛完成

环境参数：
- 观察空间：游戏画面状态
- 动作空间：球员移动和动作控制
- 奖励：得分和配合奖励
- 最大步数：由比赛设置决定

环境特色：
1. 冰球系统
   - 射门动作
   - 传球轨迹
   - 滑冰控制
   - 碰撞检测

2. 控制机制
   - 球员移动
   - 射门力度
   - 传球准度
   - 防守姿势

3. 战术元素
   - 场地利用
   - 跑位配合
   - 进攻组织
   - 防守策略

4. 评估系统
   - 进球数
   - 助攻次数
   - 防守效果
   - 整体表现

注意事项：
- 滑冰控制
- 传球时机
- 射门角度
- 防守配合
"""
"""
冰球（Ice Hockey）

```{figure} atari_ice_hockey.gif
:width: 140px
:name: ice_hockey
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import ice_hockey_v2` |
|----------------------|----------------------------------------------|
| 动作类型           | 离散                                         |
| 并行 API          | 支持                                          |
| 手动控制          | 不支持                                       |
| 智能体            | `agents= ['first_0', 'second_0']`            |
| 智能体数量        | 2                                            |
| 动作形状          | (1,)                                         |
| 动作值范围        | [0,17]                                       |
| 观察形状          | (210, 160, 3)                                |
| 观察值范围        | (0,255)                                      |


一个考验控制和时机的竞技游戏。

当你进攻时，你必须在你的两个球员之间传球（你控制持球的球员）以突破对手的防守。在防守时，你控制直接面对球的球员。双方球员都必须处理快速的控制权转换，同时绕过
对手。如果你得分，你获得 +1 奖励，你的对手获得 -1 惩罚。

[官方冰球手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=241)


#### 环境参数

环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

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


def raw_env(**kwargs):
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="ice_hockey", num_players=2, mode_num=None, env_name=name, **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
