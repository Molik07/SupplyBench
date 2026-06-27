"""
test_report.py — Phase 5 HTML Report Test
==========================================
Verifies that the HTMLReportGenerator can load results.json
and produce a clean HTML report file.

Usage:
    python test_report.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force UTF-8 stdout so formatting chars work on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json

SEPARATOR = "=" * 65
THIN_SEP  = "─" * 65

RESULTS_PATH = Path(__file__).resolve().parent / "reports" / "results.json"
OUTPUT_PATH  = Path(__file__).resolve().parent / "reports" / "report.html"


def main():
    print(f"\n{SEPARATOR}")
    print("  SupplyBench — Phase 5 HTML Report Test")
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

    # ── Step 3: Generate the HTML report ────────────────────────────── #
    print(f"\n  Generating HTML report ...")

    from report import HTMLReportGenerator

    report_gen = HTMLReportGenerator()
    report_gen.generate_from_file(str(RESULTS_PATH), str(OUTPUT_PATH))

    # ── Step 4: Verify the output file ──────────────────────────────── #
    print(f"\n  Verifying output ...")

    if OUTPUT_PATH.exists():
        size_kb = OUTPUT_PATH.stat().st_size / 1024
        print(f"  ✓ {OUTPUT_PATH.relative_to(OUTPUT_PATH.parent.parent)} exists — {size_kb:.1f} KB")
    else:
        print(f"  ✗ ERROR: {OUTPUT_PATH} was not created.")
        sys.exit(1)

    # ── Step 5: Completion ──────────────────────────────────────────── #
    print(f"\n{SEPARATOR}")
    print(f"  Phase 5 complete — Open {OUTPUT_PATH.relative_to(OUTPUT_PATH.parent.parent)} in your browser")
    print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    main()
