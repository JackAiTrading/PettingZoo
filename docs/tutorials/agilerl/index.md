# AgileRL 教程

这些教程提供了使用 [AgileRL](https://github.com/AgileRL/AgileRL) 与 PettingZoo 的入门指南。AgileRL 的多智能体算法使用 PettingZoo 并行 API，允许用户在竞争性和合作性环境中并行训练多个智能体。本教程包括以下内容：

* [DQN](DQN.md)：_通过课程学习和自我对弈训练 DQN 智能体玩四子棋_
* [MADDPG](MADDPG.md)：_训练 MADDPG 智能体玩多智能体 Atari 游戏_
* [MATD3](MATD3.md)：_训练 MATD3 智能体玩多粒子环境游戏_

## AgileRL 概述

AgileRL 是一个专注于简化强化学习模型训练的深度强化学习框架。使用[进化超参数优化](https://agilerl.readthedocs.io/en/latest/api/hpo/index.html)（HPO），AgileRL 允许用户与传统 HPO 技术相比更快、更准确地训练模型。AgileRL 的多智能体算法可以同时协调多个智能体的训练，基准测试表明，与其他强化学习库中相同算法的实现相比，在更短的时间内可以获得高达 4 倍的回报增长。

要了解更多关于 AgileRL 及其提供的其他功能，请查看[文档](https://agilerl.readthedocs.io/en/latest/)和 [GitHub 仓库](https://github.com/agilerl/agilerl)。

## 使用 PettingZoo 的示例

* [用于合作的 MADDPG：简单说话者-听众环境](https://agilerl.readthedocs.io/en/latest/multi_agent_training/index.html)


```{eval-rst}
.. figure:: test_looped.gif
   :align: center
   :height: 400px

   图1：在 6 个随机回合中训练的 MADDPG 算法的表现
```

```{toctree}
:hidden:
:caption: AgileRL

DQN
MADDPG
MATD3
```
