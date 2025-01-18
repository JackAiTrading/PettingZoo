"""生成环境演示 GIF。

这个脚本使用 asciicast2gif 工具将录制的终端会话转换为 GIF 动画。
只处理经典环境的演示。
"""

import subprocess

from pettingzoo.utils.all_modules import all_environments

# 进程列表
# procs = []

# 遍历所有环境
for name, module in all_environments.items():
    # 只处理经典环境
    if "classic" not in name:
        continue

    # 将环境名称中的斜杠替换为下划线
    nameline = name.replace("/", "_")
    # 将 JSON 格式的终端会话转换为 GIF
    proc = subprocess.run(
        ["asciicast2gif", f"gif_data/{nameline}.json", f"gifs/{nameline}.gif"]
    )
    # 并行处理控制（已注释）
    # procs.append(proc)
    # if len(procs) >= 3:
    #     for p in procs:
    #         p.wait()
    #     procs = []

# 游戏循环（未使用的代码）
# num_games = 0
# while num_games < 10000:
#     for i in range(2):
#         text = text_to_display[i]
#         # surf = font.render(text,False,(255,255,255),(0,0,0))
#         # screen.blit(surf, (0,0))
