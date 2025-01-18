"""
骑士弓箭手僵尸游戏中的武器类。

这个模块定义了游戏中的各种武器，包括剑和箭。每种武器都有其独特的物理属性和行为。
"""

import math
import os

import numpy as np
import pygame

from pettingzoo.butterfly.knights_archers_zombies.src import constants as const
from pettingzoo.butterfly.knights_archers_zombies.src.img import get_image


class Arrow(pygame.sprite.Sprite):
    """箭类。

    这个类实现了弓箭手使用的箭，包括其物理属性和飞行行为。

    属性:
        image (pygame.Surface): 箭的图像
        rect (pygame.Rect): 箭的矩形区域
        pos (list): 箭的位置 [x坐标, y坐标]
        direction (list): 箭的方向
    """

    def __init__(self, archer):
        """初始化箭。

        参数:
            archer (object): 弓箭手对象
        """
        super().__init__()
        self.archer = archer
        self.image = get_image(os.path.join("img", "arrow.png"))
        self.rect = self.image.get_rect(center=self.archer.pos)
        self.image = pygame.transform.rotate(self.image, self.archer.angle)

        self.pos = pygame.Vector2(self.archer.rect.center)
        self.direction = self.archer.direction

        # 当箭矢发射时重置弓箭手的超时时间
        archer.weapon_timeout = 0

    @property
    def vector_state(self):
        """返回箭的状态向量。

        返回一个包含箭的位置和方向的向量。
        """
        return np.array(
            [
                self.rect.x / const.SCREEN_WIDTH,
                self.rect.y / const.SCREEN_HEIGHT,
                *self.direction,
            ]
        )

    def update(self):
        """更新箭的位置和状态。

        根据当前速度和方向更新箭的位置，并检查是否超出屏幕边界。
        """
        if self.archer.alive:
            self.pos += self.direction * const.ARROW_SPEED
            self.rect.center = self.pos
        else:
            self.rect.x = -100

    @property
    def is_active(self):
        """检查箭是否处于活跃状态。

        箭处于活跃状态当且仅当其在屏幕内。
        """
        if self.rect.x < 0 or self.rect.y < 0:
            return False
        if self.rect.x > const.SCREEN_WIDTH or self.rect.y > const.SCREEN_HEIGHT:
            return False
        return True


class Sword(pygame.sprite.Sprite):
    """剑类。

    这个类实现了骑士使用的剑，包括其物理属性和攻击行为。

    属性:
        image (pygame.Surface): 剑的图像
        rect (pygame.Rect): 剑的矩形区域
        direction (list): 剑的方向
        phase (int): 剑的相位
    """

    def __init__(self, knight):
        # 这个武器实际上是一个狼牙棒，但我们在所有地方都称它为剑
        super().__init__()
        self.knight = knight
        self.image = get_image(os.path.join("img", "mace.png"))
        self.rect = self.image.get_rect(center=self.knight.rect.center)
        self.direction = self.knight.direction
        self.active = False

        # 剑的相位，从最左边部分开始
        self.phase = const.MAX_PHASE

    @property
    def vector_state(self):
        """返回剑的状态向量。

        返回一个包含剑的位置和方向的向量。
        """
        return np.array(
            [
                self.rect.x / const.SCREEN_WIDTH,
                self.rect.y / const.SCREEN_HEIGHT,
                *self.direction,
            ]
        )

    def update(self):
        """更新剑的位置和状态。

        根据当前速度和方向更新剑的位置，并检查是否超出屏幕边界。
        """
        if self.knight.action == 5:
            self.active = True

        if self.active and self.knight.alive:
            # 相位从最大值到最小值，因为它从逆时针方向计数为正
            if self.phase > const.MIN_PHASE:
                self.phase -= 1
                self.knight.attacking = True

                angle = math.radians(
                    self.knight.angle + 90 + const.SWORD_SPEED * self.phase
                )
                self.rect = self.image.get_rect(center=self.knight.rect.center)
                self.rect.x += (math.cos(angle) * (self.rect.width / 2)) + (
                    math.cos(angle) * (self.knight.rect.width / 2)
                )
                self.rect.y -= (math.sin(angle) * (self.rect.height / 2)) + (
                    math.sin(angle) * (self.knight.rect.height / 2)
                )
            else:
                self.phase = const.MAX_PHASE
                self.active = False
                self.knight.attacking = False

    @property
    def is_active(self):
        """检查剑是否处于活跃状态。

        剑处于活跃状态当且仅当其处于攻击状态。
        """
        return self.active
