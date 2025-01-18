# PettingZoo 文档

本文件夹包含 [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) 的文档。

关于如何为文档做贡献的更多信息，请参阅我们的 [CONTRIBUTING.md](https://github.com/Farama-Foundation/PettingZoo/blob/master/CONTRIBUTING.md)。

## 编辑环境页面

环境的文档位于定义环境的 Python 文件顶部。例如，国际象棋环境的文档位于 [/pettingzoo/classic/chess/chess.py](https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/chess/chess.py)。

要生成环境页面，需要执行 `docs/_scripts/gen_envs_mds.py` 脚本：

```
cd docs
python _scripts/gen_envs_mds.py
```

## 构建文档

安装所需的包和 PettingZoo：

```
pip install -e .
pip install -r docs/requirements.txt
```

一次性构建文档：

```
cd docs
make dirhtml
```

自动监视文件变化并重新构建文档：

```
cd docs
sphinx-autobuild -b dirhtml . _build
```

## 测试文档

我们使用 [pytest-markdown-docs](https://github.com/modal-labs/pytest-markdown-docs) 插件来测试文档，确保示例代码能够成功运行。要运行测试，请执行以下命令：

pytest docs --markdown-docs -m markdown-docs
