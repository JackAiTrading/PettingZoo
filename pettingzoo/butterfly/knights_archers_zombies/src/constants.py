# 视频选项
FPS = 15
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_UNITS = 15

# 僵尸速度
ZOMBIE_Y_SPEED = 5
ZOMBIE_Y_SPEED = ZOMBIE_Y_SPEED * 15.0 / FPS
if ZOMBIE_Y_SPEED % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，僵尸垂直速度 {ZOMBIE_Y_SPEED} 出现小数。"
    )
ZOMBIE_Y_SPEED = int(ZOMBIE_Y_SPEED)

ZOMBIE_X_SPEED = 30
ZOMBIE_X_SPEED = ZOMBIE_X_SPEED * 15.0 / FPS
if ZOMBIE_X_SPEED % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，僵尸水平速度 {ZOMBIE_X_SPEED} 出现小数。"
    )
ZOMBIE_X_SPEED = int(ZOMBIE_X_SPEED)

# 玩家旋转速率
PLAYER_ANG_RATE = 10
PLAYER_ANG_RATE = PLAYER_ANG_RATE * 15.0 / FPS
if PLAYER_ANG_RATE % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，角度速率 {PLAYER_ANG_RATE} 出现小数。"
    )
PLAYER_ANG_RATE = int(PLAYER_ANG_RATE)

# 弓箭手相关参数
ARCHER_X, ARCHER_Y = 400, 610

ARCHER_SPEED = 25
ARCHER_SPEED = ARCHER_SPEED * 15.0 / FPS
if ARCHER_SPEED % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，弓箭手速度 {ARCHER_SPEED} 出现小数。"
    )
ARCHER_SPEED = int(ARCHER_SPEED)

# 骑士相关参数
KNIGHT_X, KNIGHT_Y = 800, 610

KNIGHT_SPEED = 25
KNIGHT_SPEED = KNIGHT_SPEED * 15.0 / FPS
if KNIGHT_SPEED % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，骑士速度 {KNIGHT_SPEED} 出现小数。"
    )
KNIGHT_SPEED = int(KNIGHT_SPEED)

# 箭矢相关参数
ARROW_SPEED = 45
ARROW_SPEED = ARROW_SPEED * 15.0 / FPS
if ARROW_SPEED % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，箭矢速度 {ARROW_SPEED} 出现小数。"
    )
ARROW_SPEED = int(ARROW_SPEED)

# 剑相关参数
SWORD_SPEED = 20
SWORD_SPEED = SWORD_SPEED * 15.0 / FPS
if SWORD_SPEED % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，剑的速度 {SWORD_SPEED} 出现小数。"
    )
SWORD_SPEED = int(SWORD_SPEED)

MIN_PHASE = -3 / 15.0 * FPS
if MIN_PHASE % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，最小相位 {MIN_PHASE} 出现小数。"
    )
MIN_PHASE = int(MIN_PHASE)

MAX_PHASE = 3 / 15.0 * FPS
if MAX_PHASE % 1.0 != 0.0:
    raise ValueError(
        f"FPS 为 {FPS} 时，最大相位 {MAX_PHASE} 出现小数。"
    )
MAX_PHASE = int(MAX_PHASE)

ARROW_TIMEOUT = 3
SWORD_TIMEOUT = 0
