---
title: "教程：动作掩码"
---

# 教程：动作掩码

## 简介

在许多环境中，某些动作在特定时刻自然是无效的。例如，在国际象棋游戏中，如果一个兵已经在棋盘的最前面，就不可能再向前移动。在 PettingZoo 中，我们可以使用动作掩码来防止执行无效动作。

动作掩码是处理无效动作的一种更自然的方式，比起在前一个教程中我们处理撞墙的方式（让动作没有效果）更为合适。

## 代码

```{eval-rst}
.. literalinclude:: ../../../tutorials/CustomEnvironment/tutorial3_action_masking.py
   :language: python
   :caption: /custom-environment/env/custom_environment.py
```
