---
title: "CleanRL"
---

# CleanRL 教程

本教程展示如何使用 [CleanRL](https://github.com/vwxyzjn/cleanrl) 从头实现一个训练算法，并在 Pistonball 环境中训练它。

* [实现 PPO](/tutorials/cleanrl/implementing_PPO.md)：_使用简单的 PPO 实现训练智能体_

* [高级 PPO](/tutorials/cleanrl/advanced_PPO.md)：_CleanRL 的官方 PPO 示例，包含 CLI、TensorBoard 和 WandB 集成_


## CleanRL 概述

[CleanRL](https://github.com/vwxyzjn/cleanrl) 是一个轻量级、高度模块化的强化学习库，提供高质量的单文件实现，具有研究友好的特性。


更多信息请参阅[文档](https://docs.cleanrl.dev/)。

## 使用 PettingZoo 的示例：

* [PPO PettingZoo Atari 示例](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy)


## WandB 集成

CleanRL 的一个关键特性是与 [Weights & Biases](https://wandb.ai/)（WandB）的紧密集成：用于实验跟踪、超参数调优和基准测试。
[Open RL Benchmark](https://github.com/openrlbenchmark/openrlbenchmark) 允许用户查看许多任务的公共排行榜，包括智能体在训练时间步长中的表现视频。


```{figure} /_static/img/tutorials/cleanrl-wandb.png
    :alt: CleanRl 与 Weights & Biases 的集成
    :width: 80%
```


```{toctree}
:hidden:
:caption: CleanRL

implementing_PPO
advanced_PPO
```
