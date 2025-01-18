---
title: "CleanRL：高级 PPO"
---

# CleanRL：高级 PPO

本教程展示如何在 [Atari](/environments/butterfly/pistonball/) 环境（[并行](/api/parallel/)）上训练 [PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) 智能体。
这是一个完整的训练脚本，包括 CLI、日志记录以及与 [TensorBoard](https://www.tensorflow.org/tensorboard) 和 [WandB](https://wandb.ai/) 的集成，用于实验跟踪。

本教程是从 [CleanRL](https://github.com/vwxyzjn/cleanrl) 的示例中镜像的。完整的文档和实验结果可以在 [https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy) 找到。

## 环境设置
要学习本教程，你需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/CleanRL/requirements.txt
   :language: text
```

然后，使用 [AutoROM](https://github.com/Farama-Foundation/AutoROM) 安装 ROM，或使用 `rom_path` 参数指定 Atari ROM 的路径（参见[通用参数](/environments/atari/#common-parameters)）。

## 代码
以下代码应该可以正常运行。注释旨在帮助你理解如何将 PettingZoo 与 CleanRL 一起使用。如果你有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX)中询问，或在 [CleanRL 的 GitHub](https://github.com/vwxyzjn/cleanrl/issues) 上创建问题。
```{eval-rst}
.. literalinclude:: ../../../tutorials/CleanRL/cleanrl_advanced.py
   :language: python
```
