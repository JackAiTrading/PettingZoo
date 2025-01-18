---
title: "RLlib"
---

# Ray RLlib 教程

这些教程向您展示如何使用 [Ray](https://docs.ray.io/en/latest/index.html) 的 [RLlib](https://docs.ray.io/en/latest/rllib/index.html) 库在 PettingZoo 环境中训练智能体。

* [Pistonball 的 PPO 算法](/tutorials/rllib/pistonball/)：_在并行环境中训练 PPO 智能体_

* [简单扑克的 DQN 算法](/tutorials/rllib/holdem/)：_在 AEC 环境中训练 DQN 智能体_

## RLlib 概述

[RLlib](https://github.com/ray-project/ray/tree/master/rllib) 是一个工业级的开源强化学习库。
它是 [Ray](https://github.com/ray-project/ray) 的一部分，Ray 是一个用于分布式机器学习和扩展 Python 应用程序的流行库。

更多信息请参阅[文档](https://docs.ray.io/en/latest/rllib/index.html)。
 * [PettingZoo 环境](https://docs.ray.io/en/latest/rllib/rllib-env.html#pettingzoo-multi-agent-environments)
 * [已实现的算法](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html)

## 使用 PettingZoo 的示例：

### 训练：
 * [supersuit 预处理：pistonball](https://github.com/ray-project/ray/blob/master/rllib/examples/env/greyscale_env.py)
 * [简单多智能体：石头剪刀布](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)
 * [多智能体参数共享：waterworld](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_parameter_sharing.py)
 * [多智能体独立学习：waterworld](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py)
 * [多智能体 Leela 国际象棋零](https://github.com/ray-project/ray/blob/master/rllib/examples/multi-agent-leela-chess-zero.py)

[//]: # (TODO: 测试 waterworld、Leela 国际象棋零，如果未合并则向 pettingzoo 添加 PR)

### 环境：
 * [四子棋](https://github.com/ray-project/ray/blob/293fe2cb182b15499672c9cf50f79c8a9857dfb4/rllib/examples/env/pettingzoo_connect4.py)
 * [国际象棋](https://github.com/ray-project/ray/blob/293fe2cb182b15499672c9cf50f79c8a9857dfb4/rllib/examples/env/pettingzoo_chess.py)

## 架构

```{figure} https://docs.ray.io/en/latest/_images/rllib-stack.svg
    :alt: RLlib 技术栈
    :width: 80%
```

```{toctree}
:hidden:
:caption: RLlib

pistonball
holdem
```
