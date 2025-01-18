---
title: 包装器
---

# 包装器

## 使用包装器

包装器是一种环境转换，它接收一个环境作为输入，并输出一个与输入环境相似但应用了某些转换或验证的新环境。

以下包装器可以与 PettingZoo 环境一起使用：

[PettingZoo 包装器](/api/wrappers/pz_wrappers/) 包括[转换包装器](/api/wrappers/pz_wrappers#conversion-wrappers)（用于在 [AEC](/api/aec/) 和 [Parallel](/api/parallel/) API 之间转换）和一组简单的[实用工具包装器](/api/wrappers/pz_wrappers#utility-wrappers)（提供输入验证和其他方便的可重用逻辑）。

[Supersuit 包装器](/api/wrappers/supersuit_wrappers/) 包括常用的预处理功能，如帧堆叠和颜色简化，可与 PettingZoo 和 Gymnasium 兼容。

[Shimmy 兼容性包装器](/api/wrappers/shimmy_wrappers/) 允许常用的外部强化学习环境与 PettingZoo 和 Gymnasium 一起使用。


```{toctree}
:hidden:
wrappers/pz_wrappers
wrappers/supersuit_wrappers
wrappers/shimmy_wrappers
```
