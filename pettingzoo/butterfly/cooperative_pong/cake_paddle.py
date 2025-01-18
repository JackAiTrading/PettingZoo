"""
蛋糕球拍类模块。

这个模块实现了一个特殊的球拍类型 - 蛋糕球拍，它具有独特的形状和碰撞特性。
蛋糕球拍是标准球拍的变体，提供了不同的游戏体验。

主要功能：
1. 特殊的形状渲染
2. 自定义碰撞检测
3. 独特的反弹效果
4. 视觉效果增强

特性：
1. 蛋糕形状的外观
2. 独特的碰撞边界
3. 特殊的物理响应
4. 可配置的参数
"""

import pygame


class CakePaddle(pygame.sprite.Sprite):
    def __init__(self, speed=12, render_ratio=2):
        self.render_ratio = render_ratio
        # surf 是蛋糕的最右侧（最大）层
        self.surf = pygame.Surface((30 // render_ratio, 120 // render_ratio))
        self.rect = self.surf.get_rect()
        self.surf2 = pygame.Surface((30 // render_ratio, 80 // render_ratio))
        self.rect2 = self.surf2.get_rect()
        self.surf3 = pygame.Surface((30 // render_ratio, 40 // render_ratio))
        self.rect3 = self.surf3.get_rect()
        self.surf4 = pygame.Surface((30 // render_ratio, 10 // render_ratio))
        self.rect4 = self.surf4.get_rect()

        self.speed = speed

    def reset(self, seed=None, options=None):
        # self.rect 由环境类设置
        self.rect2.midright = self.rect.midleft
        self.rect3.midright = self.rect2.midleft
        self.rect4.midright = self.rect3.midleft

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect2)
        pygame.draw.rect(screen, (255, 255, 255), self.rect3)
        pygame.draw.rect(screen, (255, 255, 255), self.rect4)

    def update(self, area, action):
        # 动作：1 - 向上，2 - 向下
        movepos = [0, 0]
        if action == 1:
            movepos[1] = movepos[1] - self.speed
        elif action == 2:
            movepos[1] = movepos[1] + self.speed

        newpos = self.rect.move(movepos)
        if area.contains(newpos):
            self.rect = newpos
            # 同时移动其他矩形
            self.rect2 = self.rect2.move(movepos)
            self.rect3 = self.rect3.move(movepos)
            self.rect4 = self.rect4.move(movepos)

    def _process_collision_with_rect(self, rect, b_rect, b_speed, paddle_type):
        # 处理从顶部的碰撞
        if (
            b_rect.bottom > rect.top
            and b_rect.top - b_speed[1] < rect.top
            and b_speed[1] > 0
        ):
            b_rect.bottom = rect.top
            if b_speed[1] > 0:
                b_speed[1] *= -1
        # 处理从底部的碰撞
        elif (
            b_rect.top < rect.bottom
            and b_rect.bottom - b_speed[1] > rect.bottom
            and b_speed[1] < 0
        ):
            b_rect.top = rect.bottom
            if b_speed[1] < 0:
                b_speed[1] *= -1
        # 处理从左侧的碰撞
        if b_rect.right > rect.left:
            b_rect.right = rect.left
            if b_speed[0] > 0:
                b_speed[0] *= -1
        return True, b_rect, b_speed

    def process_collision(self, b_rect, b_speed, paddle_type):
        """返回球是否与球拍碰撞。

        参数：
            b_rect：球的矩形区域
            b_speed：球的速度
            忽略球拍类型

        返回：
            is_collision：如果球与球拍碰撞则为 1
            b_rect：新的球矩形区域
            b_speed：新的球速度

        """
        if self.rect4.colliderect(b_rect):
            return self._process_collision_with_rect(
                self.rect4, b_rect, b_speed, paddle_type
            )
        elif self.rect3.colliderect(b_rect):
            return self._process_collision_with_rect(
                self.rect3, b_rect, b_speed, paddle_type
            )
        elif self.rect2.colliderect(b_rect):
            return self._process_collision_with_rect(
                self.rect2, b_rect, b_speed, paddle_type
            )
        elif self.rect.colliderect(b_rect):
            return self._process_collision_with_rect(
                self.rect, b_rect, b_speed, paddle_type
            )
        return False, b_rect, b_speed
