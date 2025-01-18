import math
import os
import subprocess
from collections import defaultdict

from pettingzoo.utils.all_modules import all_environments


def generate_cycle_words(agents, is_classic):
    """生成循环图中的节点文字

    Args:
        agents: 智能体列表
        is_classic: 是否为经典环境

    Returns:
        节点文字列表
    """
    if is_classic:
        # 经典环境：每个智能体都显示完整名称
        words = []
        for agent in agents:
            words.append("env")
            words.append(agent)
    else:
        if len(agents) >= 10:
            # 智能体数量大于等于10时，使用范围表示
            types = []
            type_nums = defaultdict(list)
            for name in agents:
                splitname = name.split("_")
                type = splitname[0]
                type_nums[type].append(int(splitname[1]))
                if type not in types:
                    types.append(type)
            if len(types) == 1:
                # 只有一种智能体类型
                words = ["env"]
                words.append(agents[0])
                nums = type_nums[types[0]]
                type_range = f"{type}_[{nums[1]}...{nums[-1]}]"
                words.append(type_range)
            else:
                # 多种智能体类型
                words = ["env"]
                for type in types:
                    tyrange = list(range(type_nums[type][0], type_nums[type][-1] + 1))
                    assert tyrange == type_nums[type]
                    type_range = f"{type}_[{tyrange[0]}...{tyrange[-1]}]"
                    words.append(type_range)
        else:
            # 智能体数量小于10时，显示所有智能体
            words = ["env"] + agents
    return words


def generate_graphviz(words):
    """生成 Graphviz DOT 语言代码

    Args:
        words: 节点文字列表

    Returns:
        Graphviz DOT 语言代码字符串
    """
    # 计算节点宽度：最长文字长度 * 0.1 + 0.2
    max_chars = max(len(w) for w in words)
    node_width = max_chars * 0.1 + 0.2
    
    # 生成图形代码
    innards = ""  # "overlap = false;\n"
    # 设置节点样式
    innards += f'node [shape = circle,fixedsize=true,width={node_width},fontname="Segoe UI"];\n'
    
    # 生成节点
    for i, word in enumerate(words):
        # 计算节点位置（圆形布局）
        theta = 2 * math.pi * i / len(words)
        rad_const = 1.0 if len(words) <= 3 else 0.8
        rad = (len(words)) * rad_const * node_width / math.pi
        xpos = rad * math.sin(theta)
        ypos = rad * math.cos(theta)
        innards += f'a{i} [label="{word}",pos="{xpos},{ypos}!"];\n'
    
    # 生成边
    for i in range(len(words) - 1):
        innards += f"a{i} -> a{i+1};\n"
    innards += f"a{len(words)-1} -> a{0};\n"
    
    return "digraph G {\n%s\n}" % innards


# 为每个环境生成 AEC 图
for name, module in list(all_environments.items()):
    env = module.env()
    agents = env.possible_agents
    # 生成节点文字
    words = generate_cycle_words(env.possible_agents, "classic/" in name)
    # 生成 Graphviz 代码
    vis_code = generate_graphviz(words)
    # 保存代码文件
    code_path = "graphviz/" + name + ".vis"
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    with open(code_path, "w") as file:
        file.write(vis_code)
    # 生成 SVG 图像
    out_path = f"docs/assets/img/aec/{name.replace('/','_')}_aec.svg"
    cmd = ["neato", "-Tsvg", "-o", out_path, code_path]
    print(" ".join(cmd))
    subprocess.Popen(cmd)
