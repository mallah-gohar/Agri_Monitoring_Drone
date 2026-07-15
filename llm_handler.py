import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key and api_key != "your_gemini_api_key_here":
    genai.configure(api_key=api_key)

# Default model — gemini-1.5-flash is deprecated; override via GEMINI_MODEL in .env
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
]


def check_api_status():
    key = os.getenv("GEMINI_API_KEY")
    if key and key != "your_gemini_api_key_here":
        return "Connected"
    return "Not configured"


def _call_gemini(prompt, temperature=0.5):
    cfg = genai.GenerationConfig(temperature=temperature)
    models_to_try = [DEFAULT_GEMINI_MODEL] + [m for m in FALLBACK_GEMINI_MODELS if m != DEFAULT_GEMINI_MODEL]
    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name, generation_config=cfg)
            response = model.generate_content(prompt)
            return response.text
        except Exception as exc:
            last_error = exc
            err = str(exc).lower()
            if "404" in err or "not found" in err or "no longer available" in err:
                continue
            raise
    raise last_error or RuntimeError("No Gemini model available")


def generate_report(metrics, crop_type, field_name):
    if check_api_status() == "Not configured":
        return _mock_report(metrics, crop_type, field_name)

    healthy_count = metrics.get("healthy", 0)
    early_count = metrics.get("early", 0)
    severe_count = metrics.get("severe", 0)
    total = healthy_count + early_count + severe_count
    if total == 0:
        return "No cells scanned yet."

    healthy_pct = (healthy_count / total) * 100
    early_pct = (early_count / total) * 100
    severe_pct = (severe_count / total) * 100

    prompt = f"""
You are an agricultural AI assistant.
Generate a concise field health report for a farmer.
Field: {field_name} | Crop: {crop_type} | Grid: 25x25
Scan Results:
  Healthy cells  : {healthy_count} ({healthy_pct:.1f}%)
  Early disease  : {early_count}   ({early_pct:.1f}%)
  Severe disease : {severe_count}  ({severe_pct:.1f}%)
Write exactly 4 paragraphs:
  1. Overall field health summary
  2. Disease risk and likely spread pattern
  3. Specific treatment recommendations
  4. Next monitoring schedule
Plain language. No bullet points. No markdown.
"""
    try:
        return _call_gemini(prompt)
    except Exception as exc:
        return f"Error generating report: {exc}"


def get_spray_advice(metrics, crop_type):
    if check_api_status() == "Not configured":
        return _mock_spray(metrics, crop_type)

    early_count = metrics.get("early", 0)
    severe_count = metrics.get("severe", 0)

    prompt = f"""
You are a crop treatment specialist.
Crop: {crop_type}
Disease Status: {early_count} early stage cells, {severe_count} severe stage cells.
Provide a concise, 2-paragraph spraying and treatment recommendation.
Specify chemical or organic options suitable for {crop_type} with these disease levels.
Focus on containment and yield preservation.
Plain language. No bullet points. No markdown.
"""
    try:
        return _call_gemini(prompt)
    except Exception as exc:
        return f"Error generating advice: {exc}"


def query_chatbot(question, chat_history, metrics, crop_type, field_name):
    if check_api_status() == "Not configured":
        return _mock_chat(question)

    healthy_count = metrics.get("healthy", 0)
    early_count = metrics.get("early", 0)
    severe_count = metrics.get("severe", 0)
    total = healthy_count + early_count + severe_count

    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""
You are an expert AI Agronomist consulting a farmer.
Context:
  Field Name: {field_name}
  Crop Type: {crop_type}
  Scanned Grid: 25x25 (total {total} cells scanned)
  Healthy Cells: {healthy_count}
  Early Disease Cells: {early_count}
  Severe Disease Cells: {severe_count}

Conversation History:
{history_str}

User Question: {question}

Provide a helpful, professional, and practical response. Keep it concise (1-2 paragraphs).
"""
    try:
        return _call_gemini(prompt, temperature=0.7)
    except Exception as exc:
        return f"Error connecting to Gemini: {exc}"


def explain_simulation(result, metrics):
    """Gemini explanation of simulation scan results."""
    if check_api_status() == "Not configured":
        return _mock_explain("simulation", metrics, result.field_name, result.crop_type)

    prompt = f"""
You are an AI agronomist explaining drone scan results to a farmer.
Field: {result.field_name} | Crop: {result.crop_type} | Grid: 25x25
Path algorithm: {result.path_algo}
Classifier: {result.classifier_type}
Scan duration: {result.scan_duration_sec:.2f} seconds
Healthy cells: {metrics.get('healthy', 0)}
Early disease cells: {metrics.get('early', 0)}
Severe disease cells: {metrics.get('severe', 0)}
Total scanned: {metrics.get('scanned', 0)}
A* nodes explored: {result.astar_nodes_explored}

Explain in plain language:
1. What the drone found during the scan
2. What the healthy vs early vs severe counts mean for the crop
3. Which areas need urgent attention
4. What the farmer should do next
Use 4 short paragraphs. No bullet points. No markdown.
"""
    try:
        return _call_gemini(prompt)
    except Exception as exc:
        return f"Error: {exc}"


def explain_evaluation(classifier_metrics, path_metrics, result):
    """Gemini explanation of evaluation metrics and comparisons."""
    if check_api_status() == "Not configured":
        return _mock_explain("evaluation", {}, result.field_name, result.crop_type)

    rf = classifier_metrics.get("Random Forest", {})
    ndvi = classifier_metrics.get("NDVI Threshold", {})
    boustro = path_metrics.get("Boustrophedon", {})
    rw = path_metrics.get("Random Walk", {})

    prompt = f"""
You are an AI systems analyst explaining evaluation results from an agricultural drone simulator.

Classifier comparison:
  Random Forest — Accuracy: {rf.get('accuracy', 0):.1f}%, Precision: {rf.get('precision', 0):.3f}, Recall: {rf.get('recall', 0):.3f}, F1: {rf.get('f1', 0):.3f}
  NDVI Threshold — Accuracy: {ndvi.get('accuracy', 0):.1f}%, Precision: {ndvi.get('precision', 0):.3f}, Recall: {ndvi.get('recall', 0):.3f}, F1: {ndvi.get('f1', 0):.3f}

Path algorithm comparison:
  Boustrophedon — Coverage: {boustro.get('coverage', 0):.1f}%, Gaps: {boustro.get('gaps', 0)}, Path length: {boustro.get('length', 0)}, A* nodes: {boustro.get('astar_nodes', 0)}
  Random Walk — Coverage: {rw.get('coverage', 0):.1f}%, Gaps: {rw.get('gaps', 0)}, Path length: {rw.get('length', 0)}

Explain in plain language:
1. Which classifier performed better and why it matters
2. Which path algorithm is more efficient and why
3. What trade-offs exist between the two approaches
4. What you recommend for real farm monitoring
Use 4 short paragraphs. No bullet points. No markdown.
"""
    try:
        return _call_gemini(prompt)
    except Exception as exc:
        return f"Error: {exc}"


def explain_everything(result, metrics, classifier_metrics, path_metrics):
    """Comprehensive Gemini explanation of the full simulation."""
    if check_api_status() == "Not configured":
        return _mock_explain("everything", metrics, result.field_name, result.crop_type)

    rf = classifier_metrics.get("Random Forest", {})
    ndvi = classifier_metrics.get("NDVI Threshold", {})
    boustro = path_metrics.get("Boustrophedon", {})
    rw = path_metrics.get("Random Walk", {})

    prompt = f"""
You are an expert AI agronomist. Provide a complete explanation of an AgriDrone crop health simulation.

SIMULATION SETUP:
  Field: {result.field_name}
  Crop: {result.crop_type}
  Path algorithm used: {result.path_algo}
  Classifier used: {result.classifier_type}
  Disease spread steps: simulated before scan
  Scan time: {result.scan_duration_sec:.2f}s

SCAN RESULTS:
  Healthy: {metrics.get('healthy', 0)} cells
  Early disease: {metrics.get('early', 0)} cells
  Severe disease: {metrics.get('severe', 0)} cells
  Total scanned: {metrics.get('scanned', 0)} cells

CLASSIFIER COMPARISON:
  Random Forest F1: {rf.get('f1', 0):.3f} | NDVI Threshold F1: {ndvi.get('f1', 0):.3f}

PATH COMPARISON:
  Boustrophedon coverage: {boustro.get('coverage', 0):.1f}% | Random Walk coverage: {rw.get('coverage', 0):.1f}%
  A* nodes explored: {result.astar_nodes_explored}

Write a comprehensive explanation covering:
1. Overall field health and disease severity (including severe disease zones)
2. How the drone path algorithm works and why Boustrophedon vs Random Walk differ
3. How the ML classifier vs NDVI baseline detect disease
4. Practical recommendations for the farmer
5. Limitations of this simulation

Use clear paragraphs. No bullet points. No markdown. Plain farmer-friendly language.
"""
    try:
        return _call_gemini(prompt, temperature=0.6)
    except Exception as exc:
        return f"Error: {exc}"


def _mock_explain(kind, metrics, field_name, crop_type):
    healthy = metrics.get("healthy", 0)
    early = metrics.get("early", 0)
    severe = metrics.get("severe", 0)
    if kind == "everything":
        return (
            f"Complete explanation for {field_name} ({crop_type}): "
            f"The drone scanned the field and found {healthy} healthy, {early} early, and {severe} severe cells. "
            f"Boustrophedon pathing ensures systematic coverage while A* avoids obstacles. "
            f"The Random Forest classifier uses 5 spectral features while NDVI uses vegetation thresholds. "
            f"Severe zones need immediate treatment. Configure GEMINI_API_KEY in .env for live AI explanations."
        )
    if kind == "evaluation":
        return (
            "Evaluation: Random Forest typically outperforms NDVI threshold on multi-feature classification. "
            "Boustrophedon achieves higher coverage than Random Walk. (Mock — configure Gemini API for live text.)"
        )
    return (
        f"Simulation scan of {field_name}: {healthy} healthy, {early} early, {severe} severe cells detected. "
        f"Focus treatment on severe and early zones. (Mock — configure Gemini API for live explanations.)"
    )


def _mock_report(metrics, crop_type, field_name):
    healthy = metrics.get("healthy", 0)
    early = metrics.get("early", 0)
    severe = metrics.get("severe", 0)
    total = healthy + early + severe
    if total == 0:
        return "No scan data available."
    return (
        f"Field {field_name} ({crop_type}): {healthy / total * 100:.1f}% healthy, "
        f"{early / total * 100:.1f}% early disease, {severe / total * 100:.1f}% severe. "
        "Apply targeted fungicide on early zones and re-scan in 3 days. "
        "(Gemini API not configured — mock report.)"
    )


def _mock_spray(metrics, crop_type):
    early = metrics.get("early", 0)
    severe = metrics.get("severe", 0)
    return (
        f"For {crop_type}: treat {early} early and {severe} severe cells with copper-based "
        f"or Tebuconazole spray on infection boundaries. Re-assess in 48 hours. (Mock advice.)"
    )


def _mock_chat(question):
    return (
        f"Regarding '{question}': focus fungicide on early-stage fronts and reduce canopy humidity. "
        "(Gemini API not configured — mock response.)"
    )
