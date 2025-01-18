---
title: "RLlib：简单扑克的 DQN 算法"
---

# RLlib：简单扑克的 DQN 算法

本教程展示如何在 [Leduc 德州扑克](/environments/classic/leduc_holdem/) 环境（[AEC](/api/aec/)）中训练 [深度 Q 网络](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn)（DQN）智能体。

训练完成后，运行提供的代码来观看您训练的智能体与自己对战。更多信息请参阅[文档](https://docs.ray.io/en/latest/rllib/rllib-saving-and-loading-algos-and-policies.html)。

## 环境设置
要学习本教程，您需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/Ray/requirements.txt
   :language: text
```

## 代码
以下代码应该可以正常运行。注释旨在帮助您了解如何将 PettingZoo 与 RLlib 结合使用。如果您有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX) 中提问。

### 训练强化学习智能体

```{eval-rst}
.. literalinclude:: ../../../tutorials/Ray/rllib_leduc_holdem.py
   :language: python
```

### 观看训练好的强化学习智能体对战

```{eval-rst}
.. literalinclude:: ../../../tutorials/Ray/render_rllib_leduc_holdem.py
   :language: python
```
