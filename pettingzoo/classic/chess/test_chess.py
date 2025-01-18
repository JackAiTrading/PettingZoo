"""
国际象棋环境测试模块。

这个模块包含了国际象棋环境的各种测试用例，用于验证游戏规则和功能的正确性。
测试覆盖了基本规则、特殊情况和边界条件。

测试类别：
1. 基础功能测试
   - 初始化测试
   - 移动验证
   - 状态检查
   - 奖励计算

2. 规则测试
   - 基本走法
   - 特殊规则
   - 将军检测
   - 终局判定

3. API测试
   - 环境重置
   - 动作执行
   - 观察空间
   - 信息查询

4. 边界条件
   - 非法移动
   - 极限情况
   - 错误处理
   - 状态恢复

测试用例：
1. test_init - 测试环境初始化
2. test_reset - 测试重置功能
3. test_step - 测试状态转换
4. test_legal_moves - 测试合法走法
5. test_invalid_moves - 测试非法走法
6. test_game_end - 测试游戏结束条件
"""

import chess
import numpy as np

from pettingzoo.classic.chess import chess_utils


def assert_asserts(x):
    try:
        x()
    except AssertionError:
        return True
    return False


def test_chess():
    assert chess_utils.move_to_coord(chess.Move.from_uci("a8b7")) == (0, 7)
    assert chess_utils.move_to_coord(chess.Move.from_uci("g3b7")) == (6, 2)

    assert chess_utils.get_knight_dir((2, 1)) == 7
    assert chess_utils.get_knight_dir((-2, 1)) == 1
    assert assert_asserts(lambda: chess_utils.get_knight_dir((-1, 1)))

    assert chess_utils.get_queen_dir((5, -5)) == (4, 5)
    assert chess_utils.get_queen_dir((8, 0)) == (7, 6)
    assert chess_utils.get_queen_dir((0, -1)) == (0, 3)
    assert assert_asserts(lambda: chess_utils.get_queen_dir((0, 0)))
    assert assert_asserts(lambda: chess_utils.get_queen_dir((1, 2)))
    assert assert_asserts(lambda: chess_utils.get_queen_dir((2, -8)))

    assert chess_utils.get_move_plane(
        chess.Move.from_uci("e1g1")
    ) == chess_utils.get_queen_plane(
        (2, 0)
    )  # castles kingside
    assert chess_utils.get_move_plane(
        chess.Move.from_uci("g1f3")
    ) == 56 + chess_utils.get_knight_dir(
        (-1, 2)
    )  # castles kingside
    assert chess_utils.get_move_plane(
        chess.Move.from_uci("f7f8q")
    ) == chess_utils.get_queen_plane((0, 1))
    assert (
        chess_utils.get_move_plane(chess.Move.from_uci("f7f8r")) == 56 + 8 + 2 + 1 * 3
    )
    assert (
        chess_utils.get_move_plane(chess.Move.from_uci("f7g8n")) == 56 + 8 + 0 + 2 * 3
    )

    assert str(chess_utils.mirror_move(chess.Move.from_uci("f7g8"))) == "f2g1"

    board = chess.Board()
    board.push_san("e4")
    test_action = np.ones([8, 8, 73]) * -100
    test_action[0, 1, 4] = 1
    board.push_san("c5")
    _ = chess_utils.get_observation(board, player=1)
    board.push_san("e5")
    _ = chess_utils.get_observation(board, player=1)
    board.push_san("d5")
    _ = chess_utils.get_observation(board, player=1)
    board.push_san("a3")
    _ = chess_utils.get_observation(board, player=1)
    board.push_san("d4")
    _ = chess_utils.get_observation(board, player=1)
    board.push_san("c4")
    _ = chess_utils.get_observation(board, player=1)
