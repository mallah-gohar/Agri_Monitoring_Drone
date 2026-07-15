import numpy as np

import config


def calculate_path_coverage(path, grid):
    non_obstacle = int(np.sum(grid != config.CELL_OBSTACLE))
    if non_obstacle == 0:
        return 0.0, 0

    unique_visited = {p for p in set(path) if grid[p[0], p[1]] != config.CELL_OBSTACLE}
    coverage_pct = (len(unique_visited) / non_obstacle) * 100
    gaps = non_obstacle - len(unique_visited)
    return coverage_pct, gaps


def calculate_metrics(y_true, y_pred):
    y_true_bin = [1 if v in (config.CELL_EARLY, config.CELL_SEVERE) else 0 for v in y_true]
    y_pred_bin = [1 if v in (config.CELL_EARLY, config.CELL_SEVERE) else 0 for v in y_pred]

    total = len(y_true_bin)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == p)
    acc = correct / total

    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc * 100,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_classifiers(rf_predictions, ndvi_predictions, ground_truth):
    return {
        "Random Forest": calculate_metrics(ground_truth, rf_predictions),
        "NDVI Threshold": calculate_metrics(ground_truth, ndvi_predictions),
    }


def compare_paths(path_boustro, path_rw, grid, astar_nodes=0):
    cov_b, gaps_b = calculate_path_coverage(path_boustro, grid)
    cov_r, gaps_r = calculate_path_coverage(path_rw, grid)

    def efficiency(path):
        return (len(set(path)) / len(path) * 100) if path else 0.0

    return {
        "Boustrophedon": {
            "length": len(path_boustro),
            "coverage": cov_b,
            "gaps": gaps_b,
            "efficiency": efficiency(path_boustro),
            "astar_nodes": astar_nodes,
        },
        "Random Walk": {
            "length": len(path_rw),
            "coverage": cov_r,
            "gaps": gaps_r,
            "efficiency": efficiency(path_rw),
            "astar_nodes": 0,
        },
    }
