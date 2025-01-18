---
title: "LangChain"
---

# LangChain 教程

本教程提供了使用 [LangChain](https://github.com/hwchase17/langchain) 创建可以与 PettingZoo 环境交互的 LLM 智能体的示例：

* [LangChain：创建 LLM 智能体](/tutorials/langchain/langchain.md)：_使用 LangChain 创建 LLM 智能体_


## LangChain 概述

[LangChain](https://github.com/hwchase17/langchain) 是一个通过组合性来开发语言模型驱动应用程序的框架。

LangChain 旨在帮助解决六个主要领域的问题。按复杂度递增排序如下：

### 📃 LLM 和提示：

这包括提示管理、提示优化、所有 LLM 的通用接口以及使用 LLM 的常用工具。

### 🔗 链式调用：

链式调用超越了单一的 LLM 调用，涉及一系列调用（无论是对 LLM 还是其他工具的调用）。LangChain 为链式调用提供了标准接口，与其他工具的大量集成，以及用于常见应用的端到端链式调用。

### 📚 数据增强生成：

数据增强生成涉及特定类型的链式调用，这些调用首先与外部数据源交互以获取数据，用于生成步骤。示例包括长文本摘要和针对特定数据源的问答。

### 🤖 智能体：

智能体涉及 LLM 决定采取哪些行动，执行该行动，观察结果，并重复这个过程直到完成。LangChain 为智能体提供了标准接口，可供选择的智能体，以及端到端智能体的示例。

### 🧠 记忆：

记忆指的是在链式调用/智能体的调用之间保持状态。LangChain 提供了记忆的标准接口，一系列记忆实现，以及使用记忆的链式调用/智能体示例。

### 🧐 评估：

[测试版] 生成模型很难用传统指标进行评估。评估它们的一种新方法是使用语言模型本身来进行评估。LangChain 提供了一些提示/链式调用来协助这一过程。


```{toctree}
:hidden:
:caption: LangChain

langchain
```
