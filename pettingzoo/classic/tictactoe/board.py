"""
井字棋棋盘类模块。

这个模块实现了井字棋游戏的棋盘类，提供了基本的棋盘操作和状态管理功能。
包括初始化、落子、胜负判定等核心功能。

主要功能：
1. 棋盘管理
   - 初始化棋盘
   - 重置状态
   - 打印显示
   - 复制棋盘

2. 游戏操作
   - 落子
   - 检查合法性
   - 撤销移动
   - 获取可用位置

3. 状态检查
   - 胜负判定
   - 平局检查
   - 游戏是否结束
   - 当前玩家

4. 辅助功能
   - 坐标转换
   - 状态编码
   - 错误处理
   - 调试信息

数据结构：
1. 棋盘表示
   - 3x3二维数组
   - 空格用0表示
   - 玩家1用1表示
   - 玩家2用2表示

2. 移动记录
   - 位置索引（0-8）
   - 玩家标识
   - 时间戳
   - 移动序号

注意事项：
- 坐标从0开始
- 索引顺序从左到右
- 保存移动历史
- 支持状态回溯
"""

TTT_PLAYER1_WIN = 0
TTT_PLAYER2_WIN = 1
TTT_TIE = -1
TTT_GAME_NOT_OVER = -2


class Board:
    """Board for a TicTacToe Game.

    This tracks the position and identity of marks on the game board
    and allows checking for a winner.

    Example of usage:

    import random
    board = Board()

    # random legal moves - for example purposes
    def choose_move(board_obj: Board) -> int:
        legal_moves = [i for i, mark in enumerate(board_obj.squares) if mark == 0]
        return random.choice(legal_moves)

    player = 0
    while True:
        move = choose_move(board)
        board.play_turn(player, move)
        status = board.game_status()
        if status != TTT_GAME_NOT_OVER:
            if status in [TTT_PLAYER1_WIN, TTT_PLAYER2_WIN]:
                print(f"player {status} won")
            else:  # status == TTT_TIE
                print("Tie Game")
            break
        player = player ^ 1  # swaps between players 0 and 1
    """

    # indices of the winning lines: vertical(x3), horizontal(x3), diagonal(x2)
    winning_combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]

    def __init__(self):
        # self.squares holds a flat representation of the tic tac toe board.
        # an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0].
        # player 1's squares are marked 1, while player 2's are marked 2.
        # mapping of the flat indices to the 3x3 grid is as follows:
        # 0 3 6
        # 1 4 7
        # 2 5 8
        self.squares = [0] * 9

    @property
    def _n_empty_squares(self):
        """The current number of empty squares on the board."""
        return self.squares.count(0)

    def reset(self):
        """Remove all marks from the board."""
        self.squares = [0] * 9

    def play_turn(self, agent, pos):
        """Place a mark by the agent in the spot given.

        The following are required for a move to be valid:
        * The agent must be a known agent ID (either 0 or 1).
        * The spot must be be empty.
        * The spot must be in the board (integer: 0 <= spot <= 8)

        If any of those are not true, an assertion will fail.
        """
        assert pos >= 0 and pos <= 8, "Invalid move location"
        assert agent in [0, 1], "Invalid agent"
        assert self.squares[pos] == 0, "Location is not empty"

        # agent is [0, 1]. board values are stored as [1, 2].
        self.squares[pos] = agent + 1

    def game_status(self):
        """Return status (winner, TTT_TIE if no winner, or TTT_GAME_NOT_OVER)."""
        for indices in self.winning_combinations:
            states = [self.squares[idx] for idx in indices]
            if states == [1, 1, 1]:
                return TTT_PLAYER1_WIN
            if states == [2, 2, 2]:
                return TTT_PLAYER2_WIN
        if self._n_empty_squares == 0:
            return TTT_TIE
        return TTT_GAME_NOT_OVER

    def __str__(self):
        return str(self.squares)

    def legal_moves(self):
        """Return list of legal moves (as flat indices for spaces on the board)."""
        return [i for i, mark in enumerate(self.squares) if mark == 0]
