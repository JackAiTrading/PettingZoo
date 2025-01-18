---
title: "CleanRL：实现 PPO"
---

# CleanRL：实现 PPO

本教程展示如何在 [Pistonball](/environments/butterfly/pistonball/) 环境（[并行](/api/parallel/)）上训练 [PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) 智能体。

## 环境设置
要学习本教程，你需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/CleanRL/requirements.txt
   :language: text
```

## 代码
以下代码应该可以正常运行。注释旨在帮助你理解如何将 PettingZoo 与 CleanRL 一起使用。如果你有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX)中询问。
```{eval-rst}
.. literalinclude:: ../../../tutorials/CleanRL/cleanrl.py
   :language: python
```
