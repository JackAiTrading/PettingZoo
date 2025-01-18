---
title: "SB3：水世界的 PPO 算法（并行）"
---

# SB3：水世界的 PPO 算法

本教程展示如何在 [水世界](/environments/sisl/waterworld/) 环境（[并行](/api/parallel/)）中使用 [近端策略优化](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)（PPO）训练智能体。

我们使用 SuperSuit 创建向量化环境，利用多线程加速训练（参见 SB3 的[向量环境文档](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)）。

训练和评估后，此脚本将使用人类渲染启动演示游戏。训练的模型会保存到磁盘并从磁盘加载（参见 SB3 的[模型保存文档](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)）。
```{eval-rst}
.. note::

    这个环境具有离散（1 维）观察空间，所以我们使用 MLP 特征提取器。
```


## 环境设置
要学习本教程，您需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/SB3/waterworld/requirements.txt
   :language: text
```

## 代码
以下代码应该可以正常运行。注释旨在帮助您了解如何将 PettingZoo 与 SB3 结合使用。如果您有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX) 中提问。

### 训练和评估

```{eval-rst}
.. literalinclude:: ../../../tutorials/SB3/waterworld/sb3_waterworld_vector.py
   :language: python
```
