"""
ai.py
-----
Decision making for the robot arm's "brain". Operates only on the grid
representation from board.py - it never touches hardware or the camera.

Strategy:
  1. Only ever consider moves that are currently reachable (fog of war) -
     i.e. board.fog_of_war_cells(grid). The AI physically cannot "see" or
     play a floating cell, so it never generates one.
  2. Fast path: take an immediate win if one exists; otherwise block an
     immediate opponent win if one exists.
  3. Otherwise run minimax with alpha-beta pruning to a fixed depth using
     a standard windows-based heuristic (center control + counting
     2/3-in-a-rows with open ends for both sides) to pick the move that
     is best for the robot / worst for the player, i.e. "predicting the
     next grid state after its move to avoid losing".
"""

import math
import random

from board import (
    ROWS, COLS, EMPTY, PLAYER, ROBOT, WIN_LENGTH,
    apply_move, check_winner, fog_of_war_cells, is_full,
)

DEFAULT_DEPTH = 5


class RobotAI:
    def __init__(self, robot_chip=ROBOT, player_chip=PLAYER, depth=DEFAULT_DEPTH):
        self.robot_chip = robot_chip
        self.player_chip = player_chip
        self.depth = depth

    # ---------- public API ----------

    def choose_move(self, grid):
        """
        Return the column the robot wants to play. Only ever chooses among
        currently reachable columns (fog of war cells).
        """
        reachable = fog_of_war_cells(grid)
        candidate_cols = list(reachable.keys())
        if not candidate_cols:
            raise ValueError("No legal moves available; board is full.")

        # 1. Take a winning move right now if we have one.
        for col in candidate_cols:
            new_grid, _ = apply_move(grid, col, self.robot_chip)
            if check_winner(new_grid) == self.robot_chip:
                return col

        # 2. Block the player's immediate winning move if they have one.
        for col in candidate_cols:
            new_grid, _ = apply_move(grid, col, self.player_chip)
            if check_winner(new_grid) == self.player_chip:
                return col

        # 3. Otherwise, look ahead with minimax to avoid setting up a loss
        #    and to steer toward the strongest future position.
        best_score = -math.inf
        best_cols = []
        order = sorted(candidate_cols, key=lambda c: abs(c - COLS // 2))  # center-first
        for col in order:
            new_grid, _ = apply_move(grid, col, self.robot_chip)
            score = self._minimax(new_grid, self.depth - 1, -math.inf, math.inf, False)
            if score > best_score:
                best_score = score
                best_cols = [col]
            elif score == best_score:
                best_cols.append(col)

        return random.choice(best_cols)

    # ---------- minimax ----------

    def _minimax(self, grid, depth, alpha, beta, maximizing):
        reachable = fog_of_war_cells(grid)
        candidate_cols = list(reachable.keys())
        winner = check_winner(grid)

        terminal = winner is not None or not candidate_cols or depth == 0
        if terminal:
            if winner == self.robot_chip:
                return 10_000_000 + depth
            if winner == self.player_chip:
                return -10_000_000 - depth
            if not candidate_cols:
                return 0
            return self._heuristic(grid)

        if maximizing:
            value = -math.inf
            for col in candidate_cols:
                new_grid, _ = apply_move(grid, col, self.robot_chip)
                value = max(value, self._minimax(new_grid, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for col in candidate_cols:
                new_grid, _ = apply_move(grid, col, self.player_chip)
                value = min(value, self._minimax(new_grid, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # ---------- heuristic ----------

    def _heuristic(self, grid):
        score = 0

        # Center column control is disproportionately valuable in Connect 4.
        center_col = COLS // 2
        center_count = sum(1 for row in range(ROWS) if grid[row][center_col] == self.robot_chip)
        score += center_count * 6

        # Score every possible 4-in-a-row "window" on the board.
        # Horizontal
        for row in range(ROWS):
            for col in range(COLS - WIN_LENGTH + 1):
                window = [grid[row][col + i] for i in range(WIN_LENGTH)]
                score += self._score_window(window)

        # Vertical
        for col in range(COLS):
            for row in range(ROWS - WIN_LENGTH + 1):
                window = [grid[row + i][col] for i in range(WIN_LENGTH)]
                score += self._score_window(window)

        # Diagonal down-right
        for row in range(ROWS - WIN_LENGTH + 1):
            for col in range(COLS - WIN_LENGTH + 1):
                window = [grid[row + i][col + i] for i in range(WIN_LENGTH)]
                score += self._score_window(window)

        # Diagonal down-left
        for row in range(ROWS - WIN_LENGTH + 1):
            for col in range(WIN_LENGTH - 1, COLS):
                window = [grid[row + i][col - i] for i in range(WIN_LENGTH)]
                score += self._score_window(window)

        return score

    def _score_window(self, window):
        robot_count = window.count(self.robot_chip)
        player_count = window.count(self.player_chip)
        empty_count = window.count(EMPTY)

        # Mixed window (both colors present) can never become a 4-in-a-row.
        if robot_count > 0 and player_count > 0:
            return 0

        if robot_count == 4:
            return 100000
        if robot_count == 3 and empty_count == 1:
            return 100
        if robot_count == 2 and empty_count == 2:
            return 10

        if player_count == 4:
            return -100000
        if player_count == 3 and empty_count == 1:
            return -120  # weight opponent 3-in-a-rows slightly higher: prioritize blocking
        if player_count == 2 and empty_count == 2:
            return -8

        return 0
