"""Atari 游戏环境模块。

这个文件包含了所有可用的 Atari 游戏环境。
每个环境都是一个独立的模块，可以单独导入和使用。
"""

from pettingzoo.atari import (
    basketball_pong_v3,  # 篮球乒乓
    boxing_v2,  # 拳击
    combat_plane_v2,  # 战斗机
    combat_tank_v2,  # 坦克战
    double_dunk_v3,  # 双人灌篮
    entombed_competitive_v3,  # 竞争模式地下城
    entombed_cooperative_v3,  # 合作模式地下城
    flag_capture_v2,  # 夺旗
    foozpong_v3,  # 桌上足球乒乓
    ice_hockey_v2,  # 冰球
    joust_v3,  # 骑士决斗
    mario_bros_v3,  # 马里奥兄弟
    maze_craze_v3,  # 迷宫狂热
    othello_v3,  # 黑白棋
    pong_v3,  # 乒乓球
    quadrapong_v4,  # 四人乒乓
    space_invaders_v2,  # 太空入侵者
    space_war_v2,  # 太空战争
    surround_v2,  # 包围
    tennis_v3,  # 网球
    video_checkers_v4,  # 视频跳棋
    volleyball_pong_v3,  # 排球乒乓
    warlords_v3,  # 军阀
    wizard_of_wor_v3,  # 沃尔的巫师
)

# Atari 环境字典，键为环境名称，值为对应的环境类
atari_environments = {
    "atari/basketball_pong_v3": basketball_pong_v3,  # 篮球乒乓
    "atari/boxing_v2": boxing_v2,  # 拳击
    "atari/combat_tank_v2": combat_tank_v2,  # 坦克战
    "atari/combat_plane_v2": combat_plane_v2,  # 战斗机
    "atari/double_dunk_v3": double_dunk_v3,  # 双人灌篮
    "atari/entombed_competitive_v3": entombed_competitive_v3,  # 竞争模式地下城
    "atari/entombed_cooperative_v3": entombed_cooperative_v3,  # 合作模式地下城
    "atari/flag_capture_v2": flag_capture_v2,  # 夺旗
    "atari/foozpong_v3": foozpong_v3,  # 桌上足球乒乓
    "atari/joust_v3": joust_v3,  # 骑士决斗
    "atari/ice_hockey_v2": ice_hockey_v2,  # 冰球
    "atari/maze_craze_v3": maze_craze_v3,  # 迷宫狂热
    "atari/mario_bros_v3": mario_bros_v3,  # 马里奥兄弟
    "atari/othello_v3": othello_v3,  # 黑白棋
    "atari/pong_v3": pong_v3,  # 乒乓球
    "atari/quadrapong_v4": quadrapong_v4,  # 四人乒乓
    "atari/space_invaders_v2": space_invaders_v2,  # 太空入侵者
    "atari/space_war_v2": space_war_v2,  # 太空战争
    "atari/surround_v2": surround_v2,  # 包围
    "atari/tennis_v3": tennis_v3,  # 网球
    "atari/video_checkers_v4": video_checkers_v4,  # 视频跳棋
    "atari/volleyball_pong_v3": volleyball_pong_v3,  # 排球乒乓
    "atari/wizard_of_wor_v3": wizard_of_wor_v3,  # 沃尔的巫师
    "atari/warlords_v3": warlords_v3,  # 军阀
}
