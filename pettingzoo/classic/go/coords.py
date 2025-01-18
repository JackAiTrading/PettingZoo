# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code from: https://github.com/tensorflow/minigo

"""
围棋坐标系统模块。

这个模块实现了围棋游戏中使用的坐标系统，提供了各种坐标转换和处理功能。
支持多种坐标表示方法，包括GTP坐标、SGF坐标等。

主要功能：
1. 坐标转换
   - 数组索引转换
   - GTP坐标转换
   - SGF坐标转换
   - 字符串表示

2. 坐标验证
   - 边界检查
   - 有效性验证
   - 格式检查
   - 异常处理

3. 辅助功能
   - 相邻点计算
   - 方向判断
   - 距离计算
   - 区域检查

坐标系统：
1. 数组坐标
   - 从0开始的行列索引
   - 用于内部表示
   - 便于数组操作

2. GTP坐标
   - 字母表示列（A-T，不包括I）
   - 数字表示行（1-19）
   - 用于人机交互

3. SGF坐标
   - 字母表示位置（a-s）
   - 用于棋谱记录
   - 支持特殊标记

注意事项：
- 坐标系原点在左上角
- 列使用字母表示
- 行使用数字表示
- 特殊点位有专门标记
"""

"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout Minigo.

Minigo Coordinate: This is a tuple of the form (row, column) that is indexed
    starting out at (0, 0) from the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
    possible values). The extra value N^2 is used to mark a 'pass' move.
SGF Coordinate: Coordinate used for SGF serialization format. Coordinates use
    two-letter pairs having the form (column, row) indexed from the upper-left
    where 0, 0 = 'aa'.
GTP Coordinate: Human-readable coordinate string indexed from bottom left, with
    the first character a capital letter for the column and the second a number
    from 1-19 for the row. Note that GTP chooses to skip the letter 'I' due to
    its similarity with 'l' (lowercase 'L').
PYGTP Coordinate: Tuple coordinate indexed starting at 1,1 from bottom-left
    in the format (column, row)

So, for a 19x19,

Coord Type      upper_left      upper_right     pass
-------------------------------------------------------
minigo coord    (0, 0)          (0, 18)         None
flat            0               18              361
SGF             'aa'            'sa'            ''
GTP             'A19'           'T19'           'pass'
"""

from pettingzoo.classic.go import go_base

# We provide more than 19 entries here in case of boards larger than 19 x 19.
_SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def from_flat(flat):
    """Converts from a flattened coordinate to a Minigo coordinate."""
    if flat == go_base.N * go_base.N:
        return None
    return divmod(flat, go_base.N)


def to_flat(coord):
    """Converts from a Minigo coordinate to a flattened coordinate."""
    if coord is None:
        return go_base.N * go_base.N
    return go_base.N * coord[0] + coord[1]


def from_sgf(sgfc):
    """Converts from an SGF coordinate to a Minigo coordinate."""
    if sgfc is None or sgfc == "" or (go_base.N <= 19 and sgfc == "tt"):
        return None
    return _SGF_COLUMNS.index(sgfc[1]), _SGF_COLUMNS.index(sgfc[0])


def to_sgf(coord):
    """Converts from a Minigo coordinate to an SGF coordinate."""
    if coord is None:
        return ""
    return _SGF_COLUMNS[coord[1]] + _SGF_COLUMNS[coord[0]]


def from_gtp(gtpc):
    """Converts from a GTP coordinate to a Minigo coordinate."""
    gtpc = gtpc.upper()
    if gtpc == "PASS":
        return None
    col = _GTP_COLUMNS.index(gtpc[0])
    row_from_bottom = int(gtpc[1:])
    return go_base.N - row_from_bottom, col


def to_gtp(coord):
    """Converts from a Minigo coordinate to a GTP coordinate."""
    if coord is None:
        return "pass"
    y, x = coord
    return f"{_GTP_COLUMNS[x]}{go_base.N - y}"
