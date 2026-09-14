"""
board.py
--------
Pure game-state logic for Connect Four. No I/O, no ROS, no hardware.
Everything here operates on a simple 6x7 grid of ints:

    EMPTY  = 0
    PLAYER = 1
    ROBOT  = 2

Grid convention: grid[row][col], row 0 = TOP of the board, row 5 = BOTTOM
(this matches how a camera looking down/at the board would typically be
indexed, and matches how most Connect Four engines lay out the array).
Gravity means a dropped chip always lands in the LOWEST (highest row index)
empty cell of its column.
"""

from copy import deepcopy

ROWS = 6
COLS = 7

EMPTY = 0
PLAYER = 1
ROBOT = 2

WIN_LENGTH = 4


def create_empty_grid():
    """Return a fresh, empty 6x7 grid."""
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]


def clone_grid(grid):
    return deepcopy(grid)


def next_open_row(grid, col):
    """
    Return the row index a chip would land on if dropped into `col`,
    or None if the column is full.

    This is the heart of "fog of war": it is the ONLY cell in a given
    column that is currently reachable. Anything above it is not a
    legal placement (it would be a floating chip), and anything below
    it is already occupied.
    """
    if col < 0 or col >= COLS:
        return None
    for row in range(ROWS - 1, -1, -1):
        if grid[row][col] == EMPTY:
            return row
    return None


def fog_of_war_cells(grid):
    """
    Return the full set of cells that are reachable on the NEXT move,
    for either player. This is exactly the set of "next_open_row" results
    across all columns.

    Returns: dict {col: row} for every column that still has room.
    """
    reachable = {}
    for col in range(COLS):
        row = next_open_row(grid, col)
        if row is not None:
            reachable[col] = row
    return reachable


def valid_columns(grid):
    """List of columns that are not yet full."""
    return [c for c in range(COLS) if grid[0][c] == EMPTY]


def is_full(grid):
    return len(valid_columns(grid)) == 0


def apply_move(grid, col, chip):
    """
    Return a NEW grid with `chip` dropped into `col`, respecting gravity.
    Raises ValueError if the column is full. Does not mutate the input grid.
    """
    row = next_open_row(grid, col)
    if row is None:
        raise ValueError(f"Column {col} is full, illegal move.")
    new_grid = clone_grid(grid)
    new_grid[row][col] = chip
    return new_grid, row


def check_winner(grid):
    """
    Return PLAYER, ROBOT, or None depending on whether a 4-in-a-row exists.
    Checks horizontal, vertical, and both diagonals.
    """
    # Horizontal
    for row in range(ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            window = grid[row][col:col + WIN_LENGTH]
            if window[0] != EMPTY and all(v == window[0] for v in window):
                return window[0]

    # Vertical
    for col in range(COLS):
        for row in range(ROWS - WIN_LENGTH + 1):
            window = [grid[row + i][col] for i in range(WIN_LENGTH)]
            if window[0] != EMPTY and all(v == window[0] for v in window):
                return window[0]

    # Diagonal (down-right)
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(COLS - WIN_LENGTH + 1):
            window = [grid[row + i][col + i] for i in range(WIN_LENGTH)]
            if window[0] != EMPTY and all(v == window[0] for v in window):
                return window[0]

    # Diagonal (down-left)
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(WIN_LENGTH - 1, COLS):
            window = [grid[row + i][col - i] for i in range(WIN_LENGTH)]
            if window[0] != EMPTY and all(v == window[0] for v in window):
                return window[0]

    return None


def is_game_over(grid):
    return check_winner(grid) is not None or is_full(grid)


class FoulError(Exception):
    """Raised whenever an observed grid transition breaks the rules."""
    pass


def diff_cells(old_grid, new_grid):
    """Return a list of (row, col, old_val, new_val) for every cell that changed."""
    diffs = []
    for row in range(ROWS):
        for col in range(COLS):
            if old_grid[row][col] != new_grid[row][col]:
                diffs.append((row, col, old_grid[row][col], new_grid[row][col]))
    return diffs


def validate_single_legal_move(old_grid, new_grid, expected_chip):
    """
    Strictly validate that `new_grid` differs from `old_grid` by EXACTLY one
    cell, that the cell went from EMPTY -> expected_chip, and that the cell
    is a legal, gravity-respecting drop (i.e. it is the fog-of-war reachable
    cell for that column at the time of old_grid).

    Returns the (row, col) of the move on success.
    Raises FoulError on ANY violation:
        - more than one cell changed (multi-move foul)
        - zero cells changed (no move made / spurious trigger)
        - a cell changed to the wrong chip color
        - a cell changed from non-empty (chip removed/replaced)
        - a floating chip (row does not match the fog-of-war reachable row)
    """
    diffs = diff_cells(old_grid, new_grid)

    if len(diffs) == 0:
        raise FoulError("No change detected on the board; expected exactly one new chip.")

    if len(diffs) > 1:
        raise FoulError(
            f"Multiple cells changed at once ({len(diffs)} cells). "
            f"Only one move is allowed per turn."
        )

    row, col, old_val, new_val = diffs[0]

    if old_val != EMPTY:
        raise FoulError(
            f"Cell ({row},{col}) was not empty before the move (was {old_val}); "
            f"chips cannot be moved or removed."
        )

    if new_val != expected_chip:
        raise FoulError(
            f"Cell ({row},{col}) changed to chip {new_val}, "
            f"but it was chip {expected_chip}'s turn."
        )

    reachable = fog_of_war_cells(old_grid)
    expected_row = reachable.get(col)
    if expected_row is None:
        raise FoulError(f"Column {col} was already full; move is illegal.")
    if expected_row != row:
        raise FoulError(
            f"Floating chip detected at ({row},{col}); gravity requires it "
            f"to land at row {expected_row} in column {col}."
        )

    return row, col


def render(grid):
    """Human-readable board rendering for logs / console play."""
    symbols = {EMPTY: ".", PLAYER: "X", ROBOT: "O"}
    lines = []
    for row in grid:
        lines.append(" ".join(symbols[v] for v in row))
    lines.append(" ".join(str(c) for c in range(COLS)))
    return "\n".join(lines)
