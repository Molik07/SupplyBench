"""
test_report.py — HTML Report Test (with CV support)
=====================================================
Verifies that the HTMLReportGenerator can load results.json and
optionally cv_results.json, then produce a clean HTML report file.

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

RESULTS_PATH    = Path(__file__).resolve().parent / "reports" / "results.json"
CV_RESULTS_PATH = Path(__file__).resolve().parent / "reports" / "cv_results.json"
OUTPUT_PATH     = Path(__file__).resolve().parent / "reports" / "report.html"


def main():
    print(f"\n{SEPARATOR}")
    print("  SupplyBench — HTML Report Test (with CV support)")
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

    # ── Step 3: Check for CV results ────────────────────────────────── #
    has_cv = CV_RESULTS_PATH.exists()
    if has_cv:
        print(f"\n  CV results found at {CV_RESULTS_PATH.relative_to(CV_RESULTS_PATH.parent.parent)}")
        with open(CV_RESULTS_PATH, "r", encoding="utf-8") as f:
            cv_data = json.load(f)
        n_cv_models = len(cv_data.get("models", {}))
        print(f"  CV models: {n_cv_models} | Folds: {cv_data.get('n_folds', 'N/A')}")
    else:
        print(f"\n  No CV results found — report will render without CV sections")

    # ── Step 4: Generate the HTML report ────────────────────────────── #
    print(f"\n  Generating HTML report ...")

    from report import HTMLReportGenerator

    report_gen = HTMLReportGenerator()
    report_gen.generate_from_file(
        str(RESULTS_PATH),
        str(OUTPUT_PATH),
        cv_results_path=str(CV_RESULTS_PATH) if has_cv else None,
    )

    # ── Step 5: Verify the output file ──────────────────────────────── #
    print(f"\n  Verifying output ...")

    if OUTPUT_PATH.exists():
        size_kb = OUTPUT_PATH.stat().st_size / 1024
        print(f"  Report file: {OUTPUT_PATH.relative_to(OUTPUT_PATH.parent.parent)} — {size_kb:.1f} KB")

        # Read and check for CV sections in the HTML
        html_content = OUTPUT_PATH.read_text(encoding="utf-8")

        checks = {
            "Metrics table": "Metrics Comparison" in html_content,
            "Performance chart": "AVG RMSE BY MODEL" in html_content,
            "Model ranking": "Model Ranking" in html_content,
            "AI insight": "AI Insight" in html_content,
            "Benchmark suite notice": "BENCHMARK SUITE" in html_content,
        }

        if has_cv:
            checks["CV section"] = "Rolling Origin Cross Validation" in html_content
            checks["CV chart"] = "CROSS VALIDATION" in html_content
            checks["Key insight"] = "Key Insight" in html_content
            checks["Consistency rating"] = "consistency-" in html_content

        print(f"\n  Content verification:")
        all_passed = True
        for label, found in checks.items():
            status = "FOUND" if found else "MISSING"
            print(f"    {label:<28} {status}")
            if not found:
                all_passed = False
    else:
        print(f"  ERROR: {OUTPUT_PATH} was not created.")
        sys.exit(1)

    # ── Step 6: Completion ──────────────────────────────────────────── #
    print(f"\n{SEPARATOR}")
    if all_passed:
        print(f"  REPORT TEST PASSED — Open {OUTPUT_PATH.relative_to(OUTPUT_PATH.parent.parent)} in your browser")
    else:
        print(f"  REPORT TEST PARTIAL — Some sections are missing, review above")
    print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    main()
