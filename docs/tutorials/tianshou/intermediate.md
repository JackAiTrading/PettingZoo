---
title: "天授（Tianshou）：训练智能体"
---

# 天授（Tianshou）：训练智能体

本教程展示如何使用[天授（Tianshou）](https://github.com/thu-ml/tianshou)来训练一个[深度 Q 网络](https://tianshou.readthedocs.io/en/master/tutorials/dqn.html)（DQN）智能体，使其在[井字棋](/environments/classic/tictactoe/)环境中与一个[随机策略](https://tianshou.readthedocs.io/en/master/_modules/tianshou/policy/random.html)智能体对战。

## 环境设置
要学习本教程，你需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/Tianshou/requirements.txt
   :language: text
```

## 代码
以下代码应该可以直接运行。注释旨在帮助你理解如何在天授（Tianshou）中使用 PettingZoo。如果你有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX)中询问。

```{eval-rst}
.. literalinclude:: ../../../tutorials/Tianshou/2_training_agents.py
   :language: python
```
