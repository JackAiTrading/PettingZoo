"""
图像资源管理模块。

这个模块负责管理游戏中使用的所有图像资源，包括加载、缓存和处理图像。

主要功能：
1. 图像文件加载
2. 精灵表处理
3. 动画帧提取
4. 图像转换和优化

资源类型：
1. 角色精灵
   - 骑士
   - 射手
   - 僵尸

2. 武器图像
   - 剑
   - 弓箭
   - 特效

3. 背景资源
   - 地形
   - 装饰物
   - 环境元素

4. 界面元素
   - 按钮
   - 图标
   - 状态栏
"""

from os import path as os_path

import pygame


def get_image(path):
    cwd = os_path.dirname(os_path.dirname(__file__))
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc
