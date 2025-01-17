import pygame


class Paddle(pygame.sprite.Sprite):
    def __init__(self, dims, speed):
        self.surf = pygame.Surface(dims)
        self.rect = self.surf.get_rect()
        self.speed = speed

    def reset(self, seed=None, options=None):
        pass

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)

    def update(self, area, action):
        # 动作：1 - 向上，2 - 向下
        movepos = [0, 0]
        if action > 0:
            if action == 1:
                movepos[1] = movepos[1] - self.speed
            elif action == 2:
                movepos[1] = movepos[1] + self.speed

            # 确保玩家保持在屏幕内
            newpos = self.rect.move(movepos)
            if area.contains(newpos):
                self.rect = newpos

    def process_collision(self, b_rect, b_speed, paddle_type):
        """处理碰撞。

        参数：
            b_rect：球的矩形区域
            b_speed：球的速度

        返回：
            is_collision：如果球与球拍碰撞则为 1
            b_rect：新的球矩形区域
            b_speed：新的球速度
        """
        if not self.rect.colliderect(b_rect):
            return False, b_rect, b_speed
        # 处理从左侧或右侧的碰撞
        if paddle_type == 1 and b_rect.left < self.rect.right:
            b_rect.left = self.rect.right
            if b_speed[0] < 0:
                b_speed[0] *= -1
        elif paddle_type == 2 and b_rect.right > self.rect.left:
            b_rect.right = self.rect.left
            if b_speed[0] > 0:
                b_speed[0] *= -1
        # 处理从顶部的碰撞
        if (
            b_rect.bottom > self.rect.top
            and b_rect.top - b_speed[1] < self.rect.top
            and b_speed[1] > 0
        ):
            b_rect.bottom = self.rect.top
            if b_speed[1] > 0:
                b_speed[1] *= -1
        # 处理从底部的碰撞
        elif (
            b_rect.top < self.rect.bottom
            and b_rect.bottom - b_speed[1] > self.rect.bottom
            and b_speed[1] < 0
        ):
            b_rect.top = self.rect.bottom - 1
            if b_speed[1] < 0:
                b_speed[1] *= -1
        return True, b_rect, b_speed
