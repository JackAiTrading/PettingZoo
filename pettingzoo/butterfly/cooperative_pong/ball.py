"""
乒乓球类模块。

这个模块实现了乒乓球的物理特性和行为。
球体可以在游戏区域内移动，并与球拍和墙壁发生碰撞。

主要功能：
1. 球体的物理运动
2. 碰撞检测和响应
3. 速度和方向控制
4. 重置和初始化

物理特性：
1. 位置和速度
2. 弹性碰撞
3. 能量损耗
4. 角度反弹
"""

"""
球体类模块。

这个模块实现了合作乒乓游戏中球的行为，包括移动、碰撞检测和物理模拟。
"""

import numpy as np


class Ball:
    """球体类。

    这个类实现了球的物理行为，包括位置更新、速度控制和碰撞检测。

    属性:
        pos (list): 球的位置 [x, y]
        vel (list): 球的速度 [vx, vy]
        size (int): 球的大小（直径）
        speed_limit (int): 球的最大速度
        bounce_randomness (float): 球反弹时的随机性因子
    """

    def __init__(self, pos, vel, size, speed_limit=15, bounce_randomness=False):
        """初始化球体实例。

        参数:
            pos (list): 初始位置 [x, y]
            vel (list): 初始速度 [vx, vy]
            size (int): 球的大小
            speed_limit (int, 可选): 最大速度限制，默认为15
            bounce_randomness (bool, 可选): 是否启用反弹随机性，默认为False
        """
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.size = size
        self.speed_limit = speed_limit
        self.bounce_randomness = bounce_randomness
        self.contact = False

    def update(self, area, paddles):
        """更新球的位置和速度。

        处理与边界和球拍的碰撞，并更新球的位置。

        参数:
            area (list): 游戏区域的尺寸 [宽度, 高度]
            paddles (list): 球拍对象列表

        返回:
            bool: 如果球碰到左右边界则返回True，否则返回False
        """
        # 更新位置
        self.pos += self.vel

        # 检查与球拍的碰撞
        for paddle in paddles:
            if self.check_collision(paddle):
                self.bounce(paddle)

        # 检查与上下边界的碰撞
        if self.pos[1] + self.size > area[1]:  # 下边界
            self.vel[1] = -abs(self.vel[1])
            self.pos[1] = area[1] - self.size
            if self.bounce_randomness:
                self._add_bounce_randomness()

        elif self.pos[1] < 0:  # 上边界
            self.vel[1] = abs(self.vel[1])
            self.pos[1] = 0
            if self.bounce_randomness:
                self._add_bounce_randomness()

        # 检查与左右边界的碰撞
        out = False
        if self.pos[0] + self.size > area[0]:  # 右边界
            self.vel[0] = -abs(self.vel[0])
            self.pos[0] = area[0] - self.size
            out = True
            if self.bounce_randomness:
                self._add_bounce_randomness()

        elif self.pos[0] < 0:  # 左边界
            self.vel[0] = abs(self.vel[0])
            self.pos[0] = 0
            out = True
            if self.bounce_randomness:
                self._add_bounce_randomness()

        # 限制速度
        self.vel = np.clip(self.vel, -self.speed_limit, self.speed_limit)

        return out

    def check_collision(self, paddle):
        """检查是否与球拍发生碰撞。

        参数:
            paddle: 球拍对象

        返回:
            bool: 如果发生碰撞则返回True，否则返回False
        """
        # 计算球和球拍的边界框
        ball_rect = {
            "left": self.pos[0],
            "right": self.pos[0] + self.size,
            "top": self.pos[1],
            "bottom": self.pos[1] + self.size,
        }
        paddle_rect = {
            "left": paddle.pos[0],
            "right": paddle.pos[0] + paddle.size[0],
            "top": paddle.pos[1],
            "bottom": paddle.pos[1] + paddle.size[1],
        }

        # 检查边界框是否重叠
        return (
            ball_rect["left"] < paddle_rect["right"]
            and ball_rect["right"] > paddle_rect["left"]
            and ball_rect["top"] < paddle_rect["bottom"]
            and ball_rect["bottom"] > paddle_rect["top"]
        )

    def bounce(self, paddle):
        """处理与球拍的碰撞反弹。

        参数:
            paddle: 球拍对象
        """
        # 计算碰撞点相对于球拍中心的位置
        paddle_center = paddle.pos + paddle.size / 2
        ball_center = self.pos + self.size / 2
        relative_pos = ball_center - paddle_center

        # 根据碰撞点的位置调整反弹角度
        if abs(relative_pos[0]) < abs(relative_pos[1]):  # 垂直碰撞
            self.vel[1] = np.sign(relative_pos[1]) * abs(self.vel[1])
        else:  # 水平碰撞
            self.vel[0] = np.sign(relative_pos[0]) * abs(self.vel[0])

        # 添加随机性（如果启用）
        if self.bounce_randomness:
            self._add_bounce_randomness()

    def _add_bounce_randomness(self):
        """添加反弹的随机性。

        在球的速度上添加一些随机扰动，使游戏更有趣。
        """
        # 在速度上添加随机扰动
        self.vel += np.random.uniform(-0.5, 0.5, size=2)
        # 保持速度大小不变
        speed = np.linalg.norm(self.vel)
        if speed > 0:
            self.vel = self.vel / speed * self.speed_limit
