"""
标准输出捕获模块。

这个模块提供了捕获和重定向标准输出的功能。
主要用于在测试或调试时捕获程序的输出。

主要功能：
1. 捕获标准输出
2. 支持上下文管理器模式
3. 提供输出缓冲区访问
"""

import io
import sys


class capture_stdout:
    r"""Class allowing to capture stdout.

    Example:
        >>> from pettingzoo.utils.capture_stdout import capture_stdout
        >>> with capture_stdout() as var:
        ...     print("test")
        ...     data = var.getvalue()
        ...
        >>> data
        'test\n'

    标准输出捕获类。

    这个类使用上下文管理器模式，在进入时重定向标准输出到一个缓冲区，
    在退出时恢复原始的标准输出。

    用法示例：
        with capture_stdout() as var:
            print("Hello World")  # 这个输出会被捕获
            data = var.getvalue()  # 读取捕获的输出

    """

    def __init__(self):
        self.old_stdout = None

    def __enter__(self) -> io.StringIO:
        self.old_stdout = sys.stdout
        self.buff = io.StringIO()
        sys.stdout = self.buff
        return self.buff

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout
        self.buff.close()
