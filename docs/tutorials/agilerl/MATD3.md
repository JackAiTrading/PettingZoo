# AgileRL：实现 MATD3
本教程展示如何在 [simple speaker listener](https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener/) 多粒子环境中训练 [MATD3](https://agilerl.readthedocs.io/en/latest/api/algorithms/matd3.html) 智能体。

## 什么是 MATD3？
[MATD3](https://agilerl.readthedocs.io/en/latest/api/algorithms/matd3.html)（多智能体双延迟深度确定性策略梯度）扩展了 [MADDPG](https://agilerl.readthedocs.io/en/latest/api/algorithms/maddpg.html)（多智能体深度确定性策略梯度）算法，通过使用第二组评论家网络和策略网络的延迟更新，减少了多智能体领域中的过估计偏差。这使得它与 MADDPG 相比能够获得更好的性能。要了解更多关于 MATD3 的信息，请查看 AgileRL [文档](https://agilerl.readthedocs.io/en/latest/api/algorithms/matd3.html)。

### 我可以使用它吗？

|   | 动作空间 | 观察空间 |
|---|--------------|-------------------|
|离散  | ✔️           | ✔️                |
|连续   | ✔️           | ✔️                |

## 环境设置

要学习本教程，你需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/AgileRL/requirements.txt
   :language: text
```

## 代码
### 使用 MADDPG 训练多个智能体
以下代码应该可以正常运行。注释旨在帮助你理解如何将 PettingZoo 与 AgileRL 一起使用。如果你有任何问题，请随时在 [Discord 服务器](https://discord.com/invite/eB8HyTA2ux)中询问。

```{eval-rst}
.. literalinclude:: ../../../tutorials/AgileRL/agilerl_matd3.py
   :language: python
```

### 观看训练好的智能体对弈
以下代码允许你从之前的训练块中加载保存的 MATD3 算法，测试算法性能，然后将多个回合的过程可视化为 gif。
```{eval-rst}
.. literalinclude:: ../../../tutorials/AgileRL/render_agilerl_matd3.py
   :language: python
```
