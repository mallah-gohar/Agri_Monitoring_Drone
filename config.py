# config.py

GRID_ROWS = 25
GRID_COLS = 25

CROP_TYPES = ["Wheat", "Cotton", "Rice", "Sugarcane"]

# NDVI Thresholds by crop type (Healthy, Early, Severe limits)
NDVI_THRESHOLDS = {
    "Wheat":     {"healthy": 0.65, "early": 0.40, "severe": 0.20},
    "Cotton":    {"healthy": 0.70, "early": 0.45, "severe": 0.25},
    "Rice":      {"healthy": 0.60, "early": 0.35, "severe": 0.15},
    "Sugarcane": {"healthy": 0.75, "early": 0.50, "severe": 0.30},
}

# Values for the grid
CELL_HEALTHY = 0
CELL_EARLY = 1
CELL_SEVERE = 2
CELL_OBSTACLE = -1
CELL_UNSCANNED = -2

# Colors for UI heatmap (hex + matplotlib RGB tuples)
COLORS = {
    CELL_HEALTHY: "#2E7D32",    # Green
    CELL_EARLY: "#FFB300",      # Amber
    CELL_SEVERE: "#D32F2F",     # Red
    CELL_OBSTACLE: "#424242",   # Dark Grey
    CELL_UNSCANNED: "#E0E0E0"   # Light Grey
}

COLOR_RGB = {
    CELL_HEALTHY: (0.18, 0.49, 0.20),
    CELL_EARLY: (1.0, 0.70, 0.0),
    CELL_SEVERE: (0.83, 0.18, 0.18),
    CELL_OBSTACLE: (0.26, 0.26, 0.26),
    CELL_UNSCANNED: (0.88, 0.88, 0.88),
}

# Drone and path visualization colors
COLOR_DRONE = "#FFFFFF"   # White marker for drone
COLOR_PATH = "#29B6F6"    # Bright blue line for traveled path
COLOR_PATH_RGB = (0.16, 0.71, 0.96)


CLASS_LABELS = {
    CELL_HEALTHY: "Healthy",
    CELL_EARLY: "Early Disease",
    CELL_SEVERE: "Severe Disease",
    CELL_OBSTACLE: "Obstacle",
    CELL_UNSCANNED: "Unscanned"
}

FIELD_LAYOUTS = ["flat_farm", "pond_farm", "severe_outbreak", "dense_field"]

FIELD_LAYOUT_LABELS = {
    "flat_farm": "Flat Farm (Early Disease)",
    "pond_farm": "Pond Farm (Obstacles + Mixed)",
    "severe_outbreak": "Severe Outbreak Field",
    "dense_field": "Dense Severe Field",
}

FEATURE_NAMES = ["ndvi", "red_intensity", "green_intensity", "texture_variance", "moisture_index"]
