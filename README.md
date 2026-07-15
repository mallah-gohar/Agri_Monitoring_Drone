# AgriDrone — Crop Disease Detection Desktop App

A Python desktop application for simulating an agricultural monitoring drone that scans a 25×25 farm grid for crop disease. Built for the AI Lab Project with modular core logic, visual UI, explainability, and evaluation modules.

## Prerequisites

- Python 3.10+
- Optional: Google Gemini API key for live LLM reports

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API key (optional):
   Edit `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-flash-latest
   ```
   Use `gemini-flash-latest` if `gemini-1.5-flash` returns a 404 error.

3. Train the ML model:
   ```bash
   python -m ai.disease_model
   ```

4. Run the desktop app:
   ```bash
   python main.py
   ```

## Features

- **Problem Setup**: Crop type, field layout, custom JSON upload, validation
- **Search AI**: Boustrophedon coverage with A* obstacle detours vs Random Walk baseline
- **Machine Learning**: Random Forest classifier vs NDVI threshold baseline
- **Real-time Simulation**: Matplotlib grid animation with pause/step/stop/reset controls
- **Evaluation**: Accuracy, precision, recall, F1, path coverage, A* nodes explored
- **Explainability**: Feature importance and per-cell rule-based reasoning
- **AI Explanation (Gemini)**: Dedicated tab with full simulation, evaluation, and chat explanations
- **Field Layouts**: Includes **Severe Outbreak Field** with severe disease zones
- **Export**: Download scan results as CSV

## Project Structure

```
main.py              # Desktop entry point
config.py            # Constants and colors
core/                # Core logic (no UI)
ai/                  # ML training, explainability, LLM
ui/                  # PyQt6 desktop interface
data/                # Sample field JSON layouts
models/              # Saved classifier
screenshots/         # UI proof for submission
```

## Lab Deliverables

- Source code: this repository
- Sample data: `data/*.json`
- Screenshots: capture from each tab into `screenshots/`
- Report: see `REPORT.md`
