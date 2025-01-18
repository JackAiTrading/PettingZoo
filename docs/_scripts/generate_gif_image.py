import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image

from pettingzoo.utils.all_modules import all_environments


def generate_data(nameline, module):
    """
    生成数据并渲染 GIF 动画
    """
    dir = f"frames/{nameline}/"
    os.mkdir(dir)
    env = module.env(render_mode="rgb_array")
    # 初始化环境
    env.reset()
    for step in range(100):
        # 对每个智能体执行一次步骤，设置 observe=True
        for agent in env.agent_iter(env.num_agents):
            obs, rew, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]))
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)

        if env.terminations[agent] or env.truncations[agent]:
            env.reset()

        ndarray = env.render()
        # 计算缩放比例
        # tot_size = max(ndarray.shape)
        # target_size = 500
        # ratio = target_size / tot_size
        # new_shape = (int(ndarray.shape[1] * ratio), int(ndarray.shape[0] * ratio))
        im = Image.fromarray(ndarray)
        # 调整图像大小
        # im  = im.resize(new_shape, Image.ANTIALIAS)
        im.save(f"{dir}{str(step).zfill(3)}.png")
        # print(text)
    env.close()
    render_gif_image(nameline)
    # 游戏计数
    # num_games = 0
    # while num_games < 10000:
    #     for i in range(2):
    #         text = text_to_display[i]
    #         # surf = font.render(text,False,(255,255,255),(0,0,0))
    #         # screen.blit(surf, (0,0))


def render_gif_image(name):
    """
    将 PNG 序列转换为 GIF 动画
    """
    ffmpeg_command = ["convert", f"frames/{name}/*.png", f"gifs/{name}.gif"]
    print(" ".join(ffmpeg_command))
    subprocess.run(ffmpeg_command)


def render_all():
    """
    渲染所有非经典环境的 GIF 动画
    """
    for name, module in all_environments.items():
        if "classic" not in name:
            nameline = name.replace("/", "_")
            generate_data(nameline, module)
            # render_gif_image(nameline)


if __name__ == "__main__":
    name = sys.argv[1]
    if name == "all":
        render_all()
    else:
        module = all_environments[name]
        nameline = name.replace("/", "_")

        generate_data(nameline, module)
