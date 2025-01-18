---
title: "教程：环境逻辑"
---

# 教程：环境逻辑

## 简介

现在我们已经对环境仓库的结构有了基本的了解，我们可以开始思考有趣的部分 - 环境逻辑！

在本教程中，我们将创建一个双人游戏，包括一个试图逃脱的囚犯和一个试图抓住囚犯的警卫。这个游戏将在一个 7x7 的网格上进行，其中：
- 囚犯从左上角开始，
- 警卫从右下角开始，
- 逃生门随机放置在网格的中间
- 囚犯和警卫都可以在四个基本方向（上、下、左、右）移动。

## 代码

```{eval-rst}
.. literalinclude:: ../../../tutorials/CustomEnvironment/tutorial2_adding_game_logic.py
   :language: python
   :caption: /custom-environment/env/custom_environment.py
```
