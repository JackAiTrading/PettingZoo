---
title: "Stable-Baselines3"
---

# Stable-Baselines3 教程

这些教程向您展示如何使用 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)（SB3）库在 PettingZoo 环境中训练智能体。

对于具有视觉观察空间的环境，我们使用 [CNN](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.CnnPolicy) 策略，并使用 [SuperSuit](/api/wrappers/supersuit_wrappers/) 执行帧堆叠和调整大小等预处理步骤。

* [骑士-弓箭手-僵尸的 PPO 算法](/tutorials/sb3/kaz/)：_在具有视觉观察的向量化环境中使用 PPO 训练智能体_

对于非视觉环境，我们使用 [MLP](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.MlpPolicy) 策略，不执行任何预处理步骤。

* [水世界的 PPO 算法](/tutorials/sb3/waterworld/)：_在具有离散观察的向量化环境中使用 PPO 训练智能体_

* [四子棋的动作掩码 PPO 算法](/tutorials/sb3/connect_four/)：_在 AEC 环境中使用动作掩码 PPO 训练智能体_

```{eval-rst}
.. warning::

    注意：SB3 是为单智能体强化学习设计的，不直接支持多智能体算法或环境。这些教程仅用于演示目的，展示如何将 SB3 适配到 PettingZoo。
```

```{eval-rst}
.. note::

    这些教程使用带参数共享的 PPO，允许单个模型控制环境中的所有智能体。

    有关 PPO 实现细节和多智能体环境的更多信息，请参见 https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

        例如，如果有一个双人游戏，我们可以创建一个生成两个子环境的向量化环境。然后，向量化环境产生两个观察的批次，其中第一个观察来自玩家 1，第二个观察来自玩家 2。接下来，向量化环境接收两个动作的批次，并告诉游戏引擎让玩家 1 执行第一个动作，玩家 2 执行第二个动作。因此，PPO 学会在这个向量化环境中同时控制玩家 1 和玩家 2。

```


## Stable-Baselines 概述

[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)（SB3）是一个提供 [PyTorch](https://pytorch.org/) 中可靠的强化学习算法实现的库。它提供了一个干净简单的接口，让您可以访问现成的最先进的无模型强化学习算法。它只需几行代码就可以训练强化学习智能体。

更多信息，请参见 [Stable-Baselines3 v1.0 博客文章](https://araffin.github.io/post/sb3/)


```{figure} https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/_static/img/logo.png
    :alt: SB3 标志
    :width: 80%
```

```{toctree}
:hidden:
:caption: SB3

kaz
waterworld
connect_four
```
