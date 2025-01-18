---
title: "SB3：四子棋的动作掩码 PPO 算法"
---

# SB3：四子棋的动作掩码 PPO 算法

```{eval-rst}
.. warning::

   目前，本教程无法与 gymnasium>0.29.1 版本一起使用。我们正在研究修复方案，但可能需要一些时间。

```

本教程展示如何在 [四子棋](/environments/classic/chess/) 环境（[AEC](/api/aec/)）中使用可掩码的 [近端策略优化](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)（PPO）训练智能体。

它创建了一个自定义包装器，将环境转换为与 [SB3 动作掩码](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) 兼容的 [Gymnasium](https://gymnasium.farama.org/) 类环境。

训练和评估后，此脚本将使用人类渲染启动演示游戏。训练的模型会保存到磁盘并从磁盘加载（更多信息请参见 SB3 的[文档](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)）。

```{eval-rst}
.. note::

    这个环境具有带非法动作掩码的离散（1 维）观察空间，所以我们使用掩码 MLP 特征提取器。
```

```{eval-rst}
.. warning::

    SB3ActionMaskWrapper 包装器假设每个智能体的动作空间和观察空间都相同，这个假设可能不适用于自定义环境。
```


## 环境设置
要学习本教程，您需要安装下面显示的依赖项。建议使用新创建的虚拟环境以避免依赖冲突。
```{eval-rst}
.. literalinclude:: ../../../tutorials/SB3/connect_four/requirements.txt
   :language: text
```

## 代码
以下代码应该可以正常运行。注释旨在帮助您了解如何将 PettingZoo 与 SB3 结合使用。如果您有任何问题，请随时在 [Discord 服务器](https://discord.gg/nhvKkYa6qX) 中提问。

### 训练和评估

```{eval-rst}
.. literalinclude:: ../../../tutorials/SB3/connect_four/sb3_connect_four_action_mask.py
   :language: python
```

### 测试其他 PettingZoo 经典环境

以下脚本使用 [pytest](https://docs.pytest.org/en/latest/) 测试所有其他支持动作掩码的 PettingZoo 环境。

这段代码在像 [四子棋](/environments/classic/connect_four/) 这样的简单环境中可以产生不错的结果，而像 [国际象棋](/environments/classic/chess/) 或 [花火](/environments/classic/hanabi/) 这样更困难的环境可能需要更多的训练时间和超参数调整。

```{eval-rst}
.. literalinclude:: ../../../tutorials/SB3/test/test_sb3_action_mask.py
   :language: python
```
