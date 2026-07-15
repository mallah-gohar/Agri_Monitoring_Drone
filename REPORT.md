# AgriDrone — Short Lab Report

## Problem

Farmers need to detect crop disease early across large fields. Manual inspection is slow and inconsistent. AgriDrone simulates an autonomous drone scanning a 25×25 grid, classifying each cell as Healthy, Early Disease, or Severe Disease.

**Input:** Field layout JSON (obstacles, disease seeds), crop type, spread steps, algorithm settings.  
**Output:** Scanned grid map, disease metrics, evaluation comparisons, and AI-generated treatment advice.  
**Constraints:** Fixed 25×25 grid; obstacles block drone movement; disease spreads via cellular automaton.

## Method

1. Load and validate field configuration.
2. Simulate disease spread (30%/60% neighbor infection for early/severe).
3. Generate synthetic NDVI and 5-feature vectors per cell.
4. Plan drone path using Boustrophedon + A* detours or Random Walk baseline.
5. Classify each visited cell with Random Forest or NDVI thresholds.
6. Evaluate both classifiers and both path algorithms.
7. Explain predictions via feature importance, per-cell rules, and optional Gemini LLM.

## AI Components

| Component | Type | Purpose |
|-----------|------|---------|
| A* + Boustrophedon | Search/Optimization | Full field coverage with obstacle avoidance |
| Random Walk | Search baseline | Compare unstructured exploration |
| Random Forest | Machine Learning | Multi-feature disease classification |
| NDVI Threshold | Rule baseline | Compare ML vs simple vegetation index rules |
| Gemini 1.5 Flash | NLP/LLM (optional) | Field reports, spray advice, chatbot |

## Results

The desktop app shows real-time scan animation, side-by-side classifier metrics (accuracy, precision, recall, F1), and path comparison (coverage %, gaps, A* nodes explored). Boustrophedon consistently achieves higher coverage than Random Walk on obstacle fields.

## Limitations

- Synthetic NDVI/features, not real satellite imagery
- Disease spread uses simplified probabilistic rules
- Gemini requires API key and network access

## Future Improvements

- Import real multispectral drone imagery
- GPU-accelerated inference for larger grids
- Export PDF field reports
