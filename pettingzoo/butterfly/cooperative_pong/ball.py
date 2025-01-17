import numpy as np
import pygame


def get_small_random_value(randomizer):
    # 生成一个介于 [0, 1/100) 之间的小随机值
    return (1 / 100) * randomizer.random()


class Ball(pygame.sprite.Sprite):
    def __init__(self, randomizer, dims, speed, bounce_randomness=False):
        self.surf = pygame.Surface(dims)
        self.rect = self.surf.get_rect()
        self.speed_val = speed
        self.speed = [
            int(self.speed_val * np.cos(np.pi / 4)),
            int(self.speed_val * np.sin(np.pi / 4)),
        ]
        self.bounce_randomness = bounce_randomness
        self.done = False
        self.hit = False
        self.randomizer = randomizer

    def update2(self, area, p0, p1):
        # 移动球的矩形区域
        self.rect.x += self.speed[0]
        self.rect.y += self.speed[1]

        if not area.contains(self.rect):
            # 底部墙壁
            if self.rect.bottom > area.bottom:
                self.rect.bottom = area.bottom
                self.speed[1] = -self.speed[1]
            # 顶部墙壁
            elif self.rect.top < area.top:
                self.rect.top = area.top
                self.speed[1] = -self.speed[1]
            # 右侧或左侧墙壁
            else:
                return True
                self.speed[0] = -self.speed[0]

        else:
            # 球和球拍是否碰撞？
            # 添加一些随机性
            r_val = 0
            if self.bounce_randomness:
                r_val = get_small_random_value(self.randomizer)

            # 球在屏幕左半部分
            if self.rect.center[0] < area.center[0]:
                is_collision, self.rect, self.speed = p0.process_collision(
                    self.rect, self.speed, 1
                )
                if is_collision:
                    self.speed = [
                        self.speed[0] + np.sign(self.speed[0]) * r_val,
                        self.speed[1] + np.sign(self.speed[1]) * r_val,
                    ]
            # 球在右半部分
            else:
                is_collision, self.rect, self.speed = p1.process_collision(
                    self.rect, self.speed, 2
                )
                if is_collision:
                    self.speed = [
                        self.speed[0] + np.sign(self.speed[0]) * r_val,
                        self.speed[1] + np.sign(self.speed[1]) * r_val,
                    ]

        return False

    def draw(self, screen):
        # screen.blit(self.surf, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
