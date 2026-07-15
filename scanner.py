from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

import config
from core.classifier import load_classifier, predict_ndvi, predict_rf
from core.disease_spread import spread_disease
from core.grid_utils import generate_features, generate_grid, generate_ndvi_values
from core.pathfinding import build_boustrophedon_path, build_random_walk_path


@dataclass
class ScanState:
    step: int
    total: int
    drone_pos: tuple[int, int]
    scanned_grid: np.ndarray
    path_so_far: list[tuple[int, int]]
    prediction: int
    rf_prediction: int
    ndvi_prediction: int
    ground_truth: int
    ndvi: float
    features: np.ndarray
    counts: dict[int, int]


@dataclass
class SimulationResult:
    field_name: str
    crop_type: str
    path_algo: str
    classifier_type: str
    grid: np.ndarray
    ndvi_grid: np.ndarray
    features: np.ndarray
    scanned_grid: np.ndarray
    path: list[tuple[int, int]]
    path_boustro: list[tuple[int, int]]
    path_rw: list[tuple[int, int]]
    astar_nodes_explored: int
    counts: dict[int, int]
    predictions: list[int]
    rf_predictions: list[int]
    ndvi_predictions: list[int]
    ground_truth: list[int]
    ndvi_values: list[float]
    scan_duration_sec: float = 0.0


def prepare_simulation(field_data, crop_type, spread_steps, path_algo, classifier_type):
    """Prepare grid, paths, and classifier for a simulation run."""
    grid = generate_grid(field_data)
    grid = spread_disease(grid, spread_steps)
    ndvi_grid = generate_ndvi_values(grid, crop_type)
    features = generate_features(grid, ndvi_grid)

    start_pos = tuple(field_data.get("grid_configuration", {}).get("drone_start", [0, 0]))
    path_boustro, astar_nodes = build_boustrophedon_path(grid, start_pos)
    path_rw = build_random_walk_path(grid, start_pos, max_steps=len(path_boustro) or 500)
    selected_path = path_boustro if "Boustrophedon" in path_algo else path_rw

    clf = load_classifier() if "Random Forest" in classifier_type else None

    return {
        "field_data": field_data,
        "crop_type": crop_type,
        "grid": grid,
        "ndvi_grid": ndvi_grid,
        "features": features,
        "path": selected_path,
        "path_boustro": path_boustro,
        "path_rw": path_rw,
        "astar_nodes_explored": astar_nodes,
        "classifier": clf,
        "classifier_type": classifier_type,
        "path_algo": path_algo,
    }


def scan_generator(prepared):
    """Yield one ScanState per path step for real-time UI updates."""
    grid = prepared["grid"]
    ndvi_grid = prepared["ndvi_grid"]
    features = prepared["features"]
    path = prepared["path"]
    crop_type = prepared["crop_type"]
    clf = prepared["classifier"]
    use_rf = "Random Forest" in prepared["classifier_type"]

    scanned_grid = np.full_like(grid, config.CELL_UNSCANNED)
    scanned_grid[grid == config.CELL_OBSTACLE] = config.CELL_OBSTACLE

    counts = {config.CELL_HEALTHY: 0, config.CELL_EARLY: 0, config.CELL_SEVERE: 0}
    path_so_far: list[tuple[int, int]] = []
    predictions: list[int] = []
    rf_predictions: list[int] = []
    ndvi_predictions: list[int] = []
    ground_truth: list[int] = []
    ndvi_values: list[float] = []

    start_time = perf_counter()

    for step_idx, (r, c) in enumerate(path):
        actual = int(grid[r, c])
        ndvi = float(ndvi_grid[r, c])
        cell_features = features[r, c]

        rf_pred = predict_rf(clf, cell_features) if clf is not None else config.CELL_HEALTHY
        ndvi_pred = predict_ndvi(ndvi, crop_type)
        chosen = rf_pred if use_rf else ndvi_pred

        scanned_grid[r, c] = chosen
        counts[chosen] = counts.get(chosen, 0) + 1
        path_so_far.append((r, c))

        predictions.append(chosen)
        rf_predictions.append(rf_pred)
        ndvi_predictions.append(ndvi_pred)
        ground_truth.append(actual)
        ndvi_values.append(ndvi)

        yield ScanState(
            step=step_idx + 1,
            total=len(path),
            drone_pos=(r, c),
            scanned_grid=scanned_grid.copy(),
            path_so_far=path_so_far.copy(),
            prediction=chosen,
            rf_prediction=rf_pred,
            ndvi_prediction=ndvi_pred,
            ground_truth=actual,
            ndvi=ndvi,
            features=cell_features.copy(),
            counts=counts.copy(),
        )

    duration = perf_counter() - start_time
    field_data = prepared["field_data"]

    yield SimulationResult(
        field_name=field_data.get("name", "Unknown Field"),
        crop_type=crop_type,
        path_algo=prepared["path_algo"],
        classifier_type=prepared["classifier_type"],
        grid=grid,
        ndvi_grid=ndvi_grid,
        features=features,
        scanned_grid=scanned_grid,
        path=path,
        path_boustro=prepared["path_boustro"],
        path_rw=prepared["path_rw"],
        astar_nodes_explored=prepared["astar_nodes_explored"],
        counts=counts,
        predictions=predictions,
        rf_predictions=rf_predictions,
        ndvi_predictions=ndvi_predictions,
        ground_truth=ground_truth,
        ndvi_values=ndvi_values,
        scan_duration_sec=duration,
    )
