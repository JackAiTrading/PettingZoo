"""
乒乓球环境。

这个环境模拟了一个经典的双人乒乓球游戏，玩家控制球拍在场地两端
进行对抗，需要通过精准的移动和击球来得分。

主要特点：
1. 双人对战
2. 球拍控制
3. 物理碰撞
4. 技术对抗

环境规则：
1. 基本设置
   - 球拍单位
   - 乒乓球
   - 场地边界
   - 得分区域

2. 交互规则
   - 球拍移动
   - 球体反弹
   - 碰撞检测
   - 得分计算

3. 智能体行为
   - 位置调整
   - 球速预判
   - 防守反应
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
   - 反弹角度
   - 速度变化
   - 碰撞效果

2. 控制机制
   - 球拍移动
   - 击球角度
   - 速度调节
   - 位置把控

3. 战术元素
   - 场地利用
   - 节奏控制
   - 进攻变化
   - 防守站位

4. 评估系统
   - 得分情况
   - 防守成功率
   - 技术运用
   - 整体表现

注意事项：
- 位置控制
- 球速预判
- 击球时机
- 战术选择
"""
# noqa: D212, D415
"""
乒乓球游戏环境。

这个环境实现了雅达利经典游戏《乒乓球》，两名玩家在球场上进行对战。
游戏目标是用球拍击球，使对手无法接到球。

主要特点：
1. 双人对战
2. 简单直观
3. 物理引擎
4. 分数系统

游戏规则：
1. 基本设置
   - 两名玩家
   - 每人一个球拍
   - 一个乒乓球
   - 有限的场地

2. 得分规则
   - 对手接不到球得分
   - 球出界对手得分
   - 先达到指定分数获胜
   - 无平局设置

3. 球拍控制
   - 上下移动
   - 速度可调
   - 击球角度变化
   - 反弹效果

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
   - 速度变化
   - 反弹角度
   - 加速效果
   - 轨迹预测

2. 球拍特性
   - 大小固定
   - 移动速度恒定
   - 碰撞检测
   - 击球效果

3. 场地特点
   - 简单布局
   - 清晰边界
   - 对称设计
   - 无障碍物

注意事项：
- 需要良好反应
- 预判很重要
- 支持练习模式
- 可调难度
"""
"""
# 乒乓（Pong）

```{figure} atari_pong.gif
:width: 140px
:name: pong
```

此环境是<a href='..'>Atari 环境</a>的一部分。请先阅读该页面以了解基本信息。

| 导入               | `from pettingzoo.atari import pong_v3` |
|----------------------|----------------------------------------|
| 动作类型           | 离散                                   |
| 并行 API          | 支持                                    |
| 手动控制          | 不支持                                 |
| 智能体            | `agents= ['first_0', 'second_0']`      |
| 智能体数量        | 2                                      |
| 动作形状          | (1,)                                   |
| 动作值范围        | [0,5]                                  |
| 观察形状          | (210, 160, 3)                          |
| 观察值范围        | (0,255)                                |


经典的双人竞技计时游戏。

让球越过对手。

得分会给你 +1 奖励，给对手 -1 惩罚。

发球是有时间限制的：如果玩家在收到球后 2 秒内没有发球，他们会受到 -1 分的惩罚，计时器重置。这可以防止一个玩家无限期地拖延游戏，但也意味着这不再是一个纯零和游戏。

[官方视频奥运会手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

#### 环境参数

一些环境参数是所有 Atari 环境通用的，在[基础 Atari 文档](../atari)中有描述。

乒乓特有的参数如下：

``` python
pong_v3.env(num_players=2)
```

`num_players`：玩家数量（必须是 2 或 4）

### 动作空间（最小）

在任何给定回合中，智能体可以从 6 个动作中选择一个。

| 动作     | 行为    |
|:---------:|---------|
| 0         | 无操作  |
| 1         | 开火    |
| 2         | 向右移动 |
| 3         | 向左移动 |
| 4         | 向右开火 |
| 5         | 向左开火 |

### 版本历史

* v3：最小动作空间 (1.18.0)
* v2：取消动作计时器 (1.9.0)
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

avaliable_2p_versions = {
    "classic": 4,
    "two_paddles": 10,
    "soccer": 14,
    "foozpong": 19,
    "hockey": 27,
    "handball": 35,
    "volleyball": 39,
    "basketball": 45,
}
avaliable_4p_versions = {
    "classic": 6,
    "two_paddles": 11,
    "soccer": 16,
    "foozpong": 21,
    "hockey": 29,
    "quadrapong": 33,
    "handball": 37,
    "volleyball": 41,
    "basketball": 49,
}


def raw_env(num_players=2, game_version="classic", **kwargs):
    assert num_players == 2 or num_players == 4, "pong only supports 2 or 4 players"
    versions = avaliable_2p_versions if num_players == 2 else avaliable_4p_versions
    assert (
        game_version in versions
    ), f"pong version {game_version} not supported for number of players {num_players}. Available options are {list(versions)}"
    mode = versions[game_version]
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), name + "*.py")
    )
    version_num = parent_file[0].split("_")[-1].split(".")[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="pong",
        num_players=num_players,
        mode_num=mode,
        env_name=name,
        **kwargs,
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
