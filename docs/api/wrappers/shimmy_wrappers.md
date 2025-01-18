---
title: Shimmy 兼容性包装器
---

# Shimmy 兼容性包装器

[Shimmy](https://shimmy.farama.org/) 包（`pip install shimmy`）允许将常用的外部强化学习环境与 PettingZoo 和 Gymnasium 一起使用。

## 支持的多智能体环境：

### [OpenSpiel](https://shimmy.farama.org/contents/open_spiel/)
* 70 多种各类棋盘游戏的实现

### [DeepMind Control Soccer](https://shimmy.farama.org/contents/dm_multi/)
* 多智能体机器人环境，智能体团队在足球比赛中竞争。

### [DeepMind Melting Pot](https://github.com/deepmind/meltingpot)
* 多智能体强化学习的测试场景套件
* 评估对新社交情况的泛化能力：
  * 熟悉和不熟悉的个体
  * 社交互动：合作、竞争、欺骗、互惠、信任、固执
* 50 多个基础环境和 250 多个测试场景

## 用法

加载 DeepMind Control [多智能体足球游戏](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md)：

```python notest
from shimmy import DmControlMultiAgentCompatibilityV0
from dm_control.locomotion import soccer as dm_soccer

env = dm_soccer.load(team_size=2)
env = DmControlMultiAgentCompatibilityV0(env, render_mode="human")

observations, infos = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # 这里是您插入策略的地方
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

加载一个 OpenSpiel [双陆棋](https://github.com/deepmind/open_spiel/blob/master/docs/games.md#backgammon)游戏，使用 [TerminateIllegalWrapper](https://pettingzoo.farama.org/api/wrappers/pz_wrappers/#pettingzoo.utils.wrappers.TerminateIllegalWrapper) 包装：
```python notest
from shimmy import OpenSpielCompatibilityV0
from pettingzoo.utils import TerminateIllegalWrapper

env = OpenSpielCompatibilityV0(game_name="chess", render_mode=None)
env = TerminateIllegalWrapper(env, illegal_reward=-1)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(info["action_mask"])  # 这里是您插入策略的地方
    env.step(action)
    env.render()
```

加载一个 Melting Pot [矩阵中的囚徒困境](https://github.com/deepmind/meltingpot/blob/main/docs/substrate_scenario_details.md#prisoners-dilemma-in-the-matrix)基础环境：

```python notest
from shimmy import MeltingPotCompatibilityV0
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena", render_mode="human")
observations, infos = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.step(actions)
env.close()
```

更多信息，请参见 [Shimmy 文档](https://shimmy.farama.org)。

## 多智能体兼容性包装器：
```{eval-rst}
- :external:py:class:`shimmy.dm_control_multiagent_compatibility.DmControlMultiAgentCompatibilityV0`
- :external:py:class:`shimmy.openspiel_compatibility.OpenSpielCompatibilityV0`
- :external:py:class:`shimmy.meltingpot_compatibility.MeltingPotCompatibilityV0`
```

## 引用

如果您在研究中使用了这个项目，请引用：

```
@software{shimmy2022github,
  author = {{Jun Jet Tai, Mark Towers, Elliot Tower} and Jordan Terry},
  title = {Shimmy: Gymnasium and PettingZoo Wrappers for Commonly Used Environments},
  url = {https://github.com/Farama-Foundation/Shimmy},
  version = {1.0.0},
  year = {2022},
}
```
