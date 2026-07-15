import numpy as np

import config


def spread_disease(grid, steps):
    """Simulate disease spread using a cellular automaton."""
    if steps == 0:
        return grid

    current_grid = grid.copy()

    for _ in range(steps):
        next_grid = current_grid.copy()

        for r in range(config.GRID_ROWS):
            for c in range(config.GRID_COLS):
                cell_state = current_grid[r, c]
                if cell_state not in (config.CELL_EARLY, config.CELL_SEVERE):
                    continue

                prob = 0.3 if cell_state == config.CELL_EARLY else 0.6
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < config.GRID_ROWS and 0 <= nc < config.GRID_COLS:
                        if current_grid[nr, nc] == config.CELL_HEALTHY and np.random.random() < prob:
                            next_grid[nr, nc] = config.CELL_EARLY

        current_grid = next_grid

    return current_grid
