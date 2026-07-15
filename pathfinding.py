import heapq
import random

import config


def astar_detour(grid, start, goal):
    """
    A* path around obstacles.
    Returns (path, explored_nodes).
    """

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    explored = 0
    max_iterations = 1000

    while open_set and explored < max_iterations:
        explored += 1
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, explored

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < config.GRID_ROWS and 0 <= neighbor[1] < config.GRID_COLS:
                if grid[neighbor[0], neighbor[1]] != config.CELL_OBSTACLE:
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))

    return [], explored


def build_boustrophedon_path(grid, start_pos=(0, 0)):
    """Build Boustrophedon coverage path with A* detours around obstacles."""
    path = []
    current_pos = start_pos
    total_explored = 0

    for r in range(config.GRID_ROWS):
        cols = range(config.GRID_COLS) if r % 2 == 0 else range(config.GRID_COLS - 1, -1, -1)
        for c in cols:
            if grid[r, c] == config.CELL_OBSTACLE:
                continue

            target_pos = (r, c)
            dist = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
            if dist <= 1:
                path.append(target_pos)
                current_pos = target_pos
            else:
                detour, explored = astar_detour(grid, current_pos, target_pos)
                total_explored += explored
                if detour:
                    path.extend(detour)
                    current_pos = target_pos
                else:
                    path.append(target_pos)
                    current_pos = target_pos

    return path, total_explored


def build_random_walk_path(grid, start_pos=(0, 0), max_steps=500):
    """Build a random-walk baseline path for comparison."""
    path = [start_pos]
    current = start_pos
    rng = random.Random(42)

    for _ in range(max_steps - 1):
        r, c = current
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < config.GRID_ROWS and 0 <= nc < config.GRID_COLS:
                if grid[nr, nc] != config.CELL_OBSTACLE:
                    neighbors.append((nr, nc))

        if not neighbors:
            break

        unvisited = [n for n in neighbors if n not in path]
        next_cell = rng.choice(unvisited if unvisited else neighbors)
        path.append(next_cell)
        current = next_cell

    return path
