import math
import os

import numpy as np
import pygame

from pettingzoo.butterfly.knights_archers_zombies.src import constants as const
from pettingzoo.butterfly.knights_archers_zombies.src.img import get_image


class Arrow(pygame.sprite.Sprite):
    def __init__(self, archer):
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
        return np.array(
            [
                self.rect.x / const.SCREEN_WIDTH,
                self.rect.y / const.SCREEN_HEIGHT,
                *self.direction,
            ]
        )

    def update(self):
        if self.archer.alive:
            self.pos += self.direction * const.ARROW_SPEED
            self.rect.center = self.pos
        else:
            self.rect.x = -100

    @property
    def is_active(self):
        if self.rect.x < 0 or self.rect.y < 0:
            return False
        if self.rect.x > const.SCREEN_WIDTH or self.rect.y > const.SCREEN_HEIGHT:
            return False
        return True


class Sword(pygame.sprite.Sprite):
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
        return np.array(
            [
                self.rect.x / const.SCREEN_WIDTH,
                self.rect.y / const.SCREEN_HEIGHT,
                *self.direction,
            ]
        )

    def update(self):
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
        return self.active
