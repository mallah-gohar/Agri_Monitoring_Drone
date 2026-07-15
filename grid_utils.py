import json
from pathlib import Path

import numpy as np

import config


def load_field_from_json(filepath_or_buffer):
    """Load a farm layout from a file path, Path, or readable buffer."""
    if hasattr(filepath_or_buffer, "read"):
        data = json.load(filepath_or_buffer)
    else:
        with open(filepath_or_buffer, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    return data


def validate_field(field_data):
    """Validate the field configuration and raise ValueError on problems."""
    required_keys = ["name", "crop_type", "grid_configuration", "obstacles", "disease_seeds"]
    for key in required_keys:
        if key not in field_data:
            raise ValueError(f"Missing required key: {key}")

    rows = field_data["grid_configuration"].get("rows", config.GRID_ROWS)
    cols = field_data["grid_configuration"].get("cols", config.GRID_COLS)
    if rows != config.GRID_ROWS or cols != config.GRID_COLS:
        raise ValueError(f"Grid dimensions must be {config.GRID_ROWS}x{config.GRID_COLS}")

    return True


def resolve_field_path(layout_name):
    """Return absolute path for a bundled field layout."""
    return Path(__file__).resolve().parent.parent / "data" / f"{layout_name}.json"


def generate_grid(field_data):
    """Create a 25x25 NumPy array for the farm grid."""
    grid = np.full((config.GRID_ROWS, config.GRID_COLS), config.CELL_HEALTHY, dtype=int)

    for obs in field_data.get("obstacles", []):
        r, c = obs
        if 0 <= r < config.GRID_ROWS and 0 <= c < config.GRID_COLS:
            grid[r, c] = config.CELL_OBSTACLE

    for seed in field_data.get("disease_seeds", []):
        r, c = seed["cell"]
        seed_type = config.CELL_EARLY if seed["type"] == "early" else config.CELL_SEVERE
        if 0 <= r < config.GRID_ROWS and 0 <= c < config.GRID_COLS and grid[r, c] != config.CELL_OBSTACLE:
            grid[r, c] = seed_type

    return grid


def generate_ndvi_values(grid, crop_type):
    """Generate synthetic NDVI values with Gaussian noise based on cell health."""
    ndvi_grid = np.zeros_like(grid, dtype=float)
    thresholds = config.NDVI_THRESHOLDS.get(crop_type, config.NDVI_THRESHOLDS["Wheat"])

    for r in range(config.GRID_ROWS):
        for c in range(config.GRID_COLS):
            cell_state = grid[r, c]
            if cell_state == config.CELL_HEALTHY:
                val = np.random.normal(loc=thresholds["healthy"] + 0.1, scale=0.05)
                ndvi_grid[r, c] = min(1.0, max(thresholds["healthy"], val))
            elif cell_state == config.CELL_EARLY:
                val = np.random.normal(loc=thresholds["early"] + 0.05, scale=0.08)
                ndvi_grid[r, c] = min(thresholds["healthy"] - 0.01, max(thresholds["severe"] + 0.01, val))
            elif cell_state == config.CELL_SEVERE:
                val = np.random.normal(loc=thresholds["severe"] - 0.05, scale=0.06)
                ndvi_grid[r, c] = max(-1.0, min(thresholds["severe"], val))
            elif cell_state == config.CELL_OBSTACLE:
                ndvi_grid[r, c] = 0.0

    return ndvi_grid


def generate_features(grid, ndvi_grid):
    """Build a 5-feature matrix per cell."""
    features = np.zeros((config.GRID_ROWS, config.GRID_COLS, 5))

    for r in range(config.GRID_ROWS):
        for c in range(config.GRID_COLS):
            cell_state = grid[r, c]
            ndvi = ndvi_grid[r, c]

            if cell_state == config.CELL_HEALTHY:
                red = np.random.normal(0.2, 0.05)
                green = np.random.normal(0.8, 0.05)
                texture = np.random.normal(0.1, 0.02)
                moisture = np.random.normal(0.7, 0.1)
            elif cell_state == config.CELL_EARLY:
                red = np.random.normal(0.5, 0.1)
                green = np.random.normal(0.5, 0.1)
                texture = np.random.normal(0.4, 0.1)
                moisture = np.random.normal(0.4, 0.1)
            elif cell_state == config.CELL_SEVERE:
                red = np.random.normal(0.8, 0.05)
                green = np.random.normal(0.2, 0.05)
                texture = np.random.normal(0.8, 0.1)
                moisture = np.random.normal(0.1, 0.05)
            else:
                red, green, texture, moisture = 0.0, 0.0, 0.0, 0.0

            features[r, c] = [ndvi, red, green, texture, moisture]

    return features
