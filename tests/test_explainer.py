"""
test_explainer.py — Phase 4 Groq Explainer Test
================================================
Verifies that the GroqExplainer can load results.json and
generate a plain-English explanation via the Groq API.

Usage:
    python test_explainer.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 stdout so formatting chars work on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SEPARATOR = "=" * 65
THIN_SEP  = "─" * 65

RESULTS_PATH = Path(__file__).resolve().parent.parent / "reports" / "results.json"


def main():
    print(f"\n{SEPARATOR}")
    print("  SupplyBench — Phase 4 Groq Explainer Test")
    print(SEPARATOR)

    # ── Step 1: Check if results.json exists ────────────────────────── #
    if not RESULTS_PATH.exists():
        print(f"\n  ERROR: {RESULTS_PATH} not found.")
        print("  Please run 'python cli.py run' first to generate benchmark results.")
        sys.exit(1)

    # ── Step 2: Load results and print confirmation ─────────────────── #
    print(f"\n  Loading results from {RESULTS_PATH.relative_to(RESULTS_PATH.parent.parent)} ... ", end="")
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    print("done")

    dataset = results.get("dataset", "unknown")
    store = results.get("store", "unknown")
    items = results.get("items_evaluated", 0)
    print(f"  Dataset: {dataset} | Store: {store} | Items: {items}")

    # ── Step 3: Call the Groq API ───────────────────────────────────── #
    print(f"\n  Calling Groq API ...")

    from explainer import GroqExplainer

    explainer = GroqExplainer()
    explanation = explainer.explain_from_file(str(RESULTS_PATH))

    # ── Step 4: Print the explanation ───────────────────────────────── #
    print(f"\n{THIN_SEP}")
    print("  AI Explanation")
    print(THIN_SEP)
    print()
    print(f"  {explanation}")
    print()
    print(THIN_SEP)

    # ── Step 5: Confirmation ────────────────────────────────────────── #
    print(f"\n{SEPARATOR}")
    print("  Phase 4 complete — Groq explainer working")
    print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    main()
