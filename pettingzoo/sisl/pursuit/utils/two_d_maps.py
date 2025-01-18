import numpy as np
from scipy.ndimage import zoom


def rectangle_map(xs, ys, xb=0.3, yb=0.2):
    """返回一个2D'地图'，中间有一个矩形建筑。

    地图是一个2D numpy数组
    xb和yb是每个维度的缓冲区，表示在每一侧留出的地图比例
    """
    rmap = np.zeros((xs, ys), dtype=np.int32)
    for i in range(xs):
        for j in range(ys):
            # 我们是否在x维度的矩形内？
            if (float(i) / xs) > xb and (float(i) / xs) < (1.0 - xb):
                # 我们是否在y维度的矩形内？
                if (float(j) / ys) > yb and (float(j) / ys) < (1.0 - yb):
                    rmap[i, j] = -1  # -1是建筑物像素标志
    return rmap


def complex_map(xs, ys):
    """返回一个带有四个不同障碍物的2D'地图'。

    地图是一个2D numpy数组
    """
    cmap = np.zeros((xs, ys), dtype=np.int32)
    cmap = add_rectangle(cmap, xc=0.8, yc=0.5, xl=0.1, yl=0.8)
    cmap = add_rectangle(cmap, xc=0.4, yc=0.8, xl=0.5, yl=0.2)
    cmap = add_rectangle(cmap, xc=0.5, yc=0.5, xl=0.4, yl=0.2)
    cmap = add_rectangle(cmap, xc=0.3, yc=0.1, xl=0.5, yl=0.1)
    cmap = add_rectangle(cmap, xc=0.1, yc=0.3, xl=0.1, yl=0.5)
    return cmap


def gen_map(
    xs,
    ys,
    n_obs,
    randomizer,
    center_bounds=[0.0, 1.0],
    length_bounds=[0.1, 0.5],
    gmap=None,
):
    """生成一个随机地图

    参数：
        xs, ys：地图尺寸
        n_obs：障碍物数量
        randomizer：随机数生成器
        center_bounds：中心点范围
        length_bounds：长度范围
        gmap：现有地图（如果为None则创建新地图）
    """
    cl, cu = center_bounds
    ll, lu = length_bounds
    if gmap is None:
        gmap = np.zeros((xs, ys), dtype=np.int32)
    for _ in range(n_obs):
        xc = randomizer.uniform(cl, cu)
        yc = randomizer.uniform(cl, cu)
        xl = randomizer.uniform(ll, lu)
        yl = randomizer.uniform(ll, lu)
        gmap = add_rectangle(gmap, xc=xc, yc=yc, xl=xl, yl=yl)
    return gmap


def multi_scale_map(
    xs,
    ys,
    randomizer,
    scales=[(3, [0.2, 0.3]), (10, [0.1, 0.2]), (30, [0.05, 0.1]), (150, [0.01, 0.05])],
):
    """生成多尺度地图

    参数：
        xs, ys：地图尺寸
        randomizer：随机数生成器
        scales：尺度列表，每个元素为(障碍物数量, [最小长度, 最大长度])
    """
    gmap = np.zeros((xs, ys), dtype=np.int32)
    for scale in scales:
        n, lb = scale
        gmap = gen_map(xs, ys, n, randomizer, length_bounds=lb, gmap=gmap)
    return gmap


def add_rectangle(input_map, xc, yc, xl, yl):
    """向输入地图添加一个矩形。

    参数：
        input_map：输入地图
        xc, yc：矩形中心点（相对于地图的归一化坐标）
        xl, yl：矩形尺寸（相对于地图的归一化尺寸）
    """
    assert len(input_map.shape) == 2, "input_map必须是numpy矩阵"

    xs, ys = input_map.shape
    xcc, ycc = int(round(xs * xc)), int(round(ys * yc))
    xll, yll = int(round(xs * xl)), int(round(ys * yl))
    if xll <= 1:
        x_lbound, x_upbound = xcc, xcc + 1
    else:
        x_lbound, x_upbound = xcc - xll / 2, xcc + xll / 2
    if yll <= 1:
        y_lbound, y_upbound = ycc, ycc + 1
    else:
        y_lbound, y_upbound = ycc - yll / 2, ycc + yll / 2

    # assert x_lbound >= 0 and x_upbound < xs, "无效的矩形配置，x超出边界"
    # assert y_lbound >= 0 and y_upbound < ys, "无效的矩形配置，y超出边界"

    x_lbound, x_upbound = np.clip([x_lbound, x_upbound], 0, xs)
    y_lbound, y_upbound = np.clip([y_lbound, y_upbound], 0, ys)

    for i in range(x_lbound, x_upbound):
        for j in range(y_lbound, y_upbound):
            input_map[j, i] = -1
    return input_map


def resize(scale, old_mats):
    """调整矩阵尺寸

    参数：
        scale：缩放比例
        old_mats：原始矩阵列表
    """
    new_mats = []
    for mat in old_mats:
        new_mats.append(zoom(mat, scale, order=0))
    return np.array(new_mats)


def simple_soccer_map(xs=6, ys=9):
    """生成简单的足球场地图

    参数：
        xs：x轴尺寸（必须为偶数）
        ys：y轴尺寸
    """
    assert xs % 2 == 0, "xs必须是偶数"
    smap = np.zeros((xs, ys), dtype=np.int32)
    smap[0 : xs / 2 - 1, 0] = -1
    smap[xs / 2 + 1 : xs, 0] = -1
    smap[0 : xs / 2 - 1, ys - 1] = -1
    smap[xs / 2 + 1 : xs, ys - 1] = -1
    return smap


def cross_map(xs, ys):
    """生成十字形地图"""
    pass
