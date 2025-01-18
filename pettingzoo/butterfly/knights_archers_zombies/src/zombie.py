"""
骑士弓箭手僵尸游戏中的僵尸类。

这个模块定义了游戏中的僵尸敌人，包括其物理属性和行为。
"""

import os

import numpy as np
import pygame

from pettingzoo.butterfly.knights_archers_zombies.src import constants as const
from pettingzoo.butterfly.knights_archers_zombies.src.img import get_image


class Zombie(pygame.sprite.Sprite):
    """僵尸类。

    这个类实现了游戏中的僵尸敌人，包括其物理属性和移动行为。

    属性:
        image (pygame.Surface): 僵尸的图像
        rect (pygame.Rect): 僵尸的矩形区域
        x_lims (list): 僵尸的 x 轴限制
        randomizer (random.Random): 随机数生成器，用于生成僵尸的移动方向
    """

    def __init__(self, randomizer):
        """初始化僵尸。

        参数:
            randomizer (random.Random): 随机数生成器，用于生成僵尸的移动方向
        """
        super().__init__()
        self.image = get_image(os.path.join("img", "zombie.png"))
        self.rect = self.image.get_rect(center=(50, 50))
        self.randomizer = randomizer

        self.x_lims = [const.SCREEN_UNITS, const.SCREEN_WIDTH - const.SCREEN_UNITS]

    @property
    def vector_state(self):
        """返回僵尸的状态向量。

        返回:
            ndarray: 包含僵尸位置的向量
        """
        return np.array(
            [
                self.rect.x / const.SCREEN_WIDTH,
                self.rect.y / const.SCREEN_HEIGHT,
                0.0,
                1.0,
            ]
        )

    def update(self):
        """更新僵尸的位置和状态。

        根据当前速度和方向更新僵尸的位置，并确保僵尸朝向正确的方向。
        """
        rand_x = self.randomizer.integers(0, 10)

        self.rect.y += const.ZOMBIE_Y_SPEED

        # 在 X-Y 方向上摇摆
        if self.rect.y % const.SCREEN_UNITS == 0:
            if self.rect.x > self.x_lims[0] and self.rect.x < self.x_lims[1]:
                if rand_x in [1, 3, 6]:
                    self.rect.x += const.ZOMBIE_X_SPEED
                elif rand_x in [2, 4, 5, 8]:
                    self.rect.x -= const.ZOMBIE_X_SPEED

            # 将僵尸带回屏幕
            else:
                if self.rect.x <= self.x_lims[0]:
                    self.rect.x += 2 * const.ZOMBIE_X_SPEED
                elif self.rect.x >= self.x_lims[1]:
                    self.rect.x -= 2 * const.ZOMBIE_X_SPEED

        # 限制在屏幕内
        self.rect.x = max(min(self.rect.x, const.SCREEN_WIDTH - 100), 100)
