from __future__ import annotations
from typing import List, Optional
import numpy as np


class Agent:

    def __init__(self, seed: Optional[int] = None, depth: int = 3):
        self.depth = depth
        self.rng = np.random.default_rng(seed)
        self.cache = {}

        # Snake pattern weights
        self.base_weights = np.array([
            [15, 14, 13, 12],
            [ 8,  9, 10, 11],
            [ 7,  6,  5,  4],
            [ 0,  1,  2,  3]
        ], dtype=np.float32)

    # ====================================================
    # MÉTODO PRINCIPAL
    # ====================================================
    def act(self, board: np.ndarray, legal_actions: List[str]) -> str:

        if not legal_actions:
            return "up"

        self.cache.clear()

        best_score = -float("inf")
        best_action = legal_actions[0]

        for action in legal_actions:
            new_board, reward, moved = self.simulate_move(board, action)
            if not moved:
                continue

            score = reward + self.expectimax(new_board, self.depth - 1, False)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ====================================================
    # EXPECTIMAX
    # ====================================================
    def expectimax(self, board, depth, player_turn):

        key = (tuple(board.flatten()), depth, player_turn)
        if key in self.cache:
            return self.cache[key]

        if depth == 0:
            val = self.evaluate(board)
            self.cache[key] = val
            return val

        if player_turn:
            best = -float("inf")
            for action in ["up", "down", "left", "right"]:
                new_board, reward, moved = self.simulate_move(board, action)
                if not moved:
                    continue

                val = reward + self.expectimax(new_board, depth - 1, False)
                best = max(best, val)

            if best == -float("inf"):
                best = self.evaluate(board)

            self.cache[key] = best
            return best

        else:
            empties = list(zip(*np.where(board == 0)))
            if not empties:
                val = self.evaluate(board)
                self.cache[key] = val
                return val

            total = 0.0

            for r, c in empties:
                for value, prob in [(2, 0.9), (4, 0.1)]:
                    new_board = board.copy()
                    new_board[r, c] = value
                    total += prob * self.expectimax(new_board, depth - 1, True)

            val = total / len(empties)
            self.cache[key] = val
            return val

    # ====================================================
    # SIMULADOR DE MOVIMIENTO
    # ====================================================
    def simulate_move(self, board, action):
        board_copy = board.copy()
        size = board.shape[0]
        reward = 0
        moved_any = False

        def merge_line(line):
            nonlocal reward
            nonzero = line[line != 0].tolist()
            merged = []
            i = 0
            while i < len(nonzero):
                if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                    v = nonzero[i] * 2
                    merged.append(v)
                    reward += v
                    i += 2
                else:
                    merged.append(nonzero[i])
                    i += 1
            new_line = np.zeros_like(line)
            new_line[:len(merged)] = merged
            return new_line

        if action in ("left", "right"):
            for i in range(size):
                row = board_copy[i, :]
                if action == "right":
                    row = row[::-1]
                new_row = merge_line(row)
                if action == "right":
                    new_row = new_row[::-1]
                if not np.array_equal(new_row, board_copy[i, :]):
                    moved_any = True
                board_copy[i, :] = new_row
        else:
            for j in range(size):
                col = board_copy[:, j]
                if action == "down":
                    col = col[::-1]
                new_col = merge_line(col)
                if action == "down":
                    new_col = new_col[::-1]
                if not np.array_equal(new_col, board_copy[:, j]):
                    moved_any = True
                board_copy[:, j] = new_col

        return board_copy, reward, moved_any

    # ====================================================
    # FUNCIÓN DE EVALUACIÓN OPTIMIZADA
    # ====================================================
    def evaluate(self, board):

        empty = np.sum(board == 0)

        # log2 seguro (sin warning)
        log_board = np.zeros_like(board, dtype=np.float32)
        mask = board > 0
        log_board[mask] = np.log2(board[mask])

        # Snake pattern (evaluamos las 4 rotaciones)
        snake_score = 0.0
        for k in range(4):
            rotated = np.rot90(log_board, k)
            weights = np.rot90(self.base_weights, k)
            snake_score = max(snake_score, np.sum(rotated * weights))

        # Monotonicidad
        mono = 0.0
        for i in range(4):
            mono -= np.sum(np.abs(log_board[i, :-1] - log_board[i, 1:]))
            mono -= np.sum(np.abs(log_board[:-1, i] - log_board[1:, i]))

        # Potencial de merges
        merge_potential = 0.0
        for i in range(4):
            for j in range(3):
                if board[i, j] == board[i, j+1] and board[i, j] != 0:
                    merge_potential += np.log2(board[i, j])
                if board[j, i] == board[j+1, i] and board[j, i] != 0:
                    merge_potential += np.log2(board[j, i])

        # Corner bonus
        max_tile = board.max()
        corner_bonus = 0.0
        corners = [board[0,0], board[0,-1], board[-1,0], board[-1,-1]]
        if max_tile in corners and max_tile > 0:
            corner_bonus = np.log2(max_tile) * 40

        return (
            snake_score * 12
            + empty * 250
            + mono * 1.2
            + merge_potential * 80
            + corner_bonus
        )