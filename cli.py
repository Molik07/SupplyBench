"""
cli.py — SupplyBench Command-Line Interface
============================================
The single entry point for the entire SupplyBench benchmarking tool.
Orchestrates data loading, model fitting, metric evaluation and
results output, all driven by config.yaml.

Usage:
    python cli.py run
    python cli.py run --item HOBBIES_1_001
"""

import sys

# Force UTF-8 stdout so box-drawing chars work on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import os
from pathlib import Path
from collections import OrderedDict

import click
import yaml
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).resolve().parent / ".env")

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loader import load_m5_data
from baselines import (
    NaiveForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    ARIMAForecaster,
    ProphetForecaster,
)
from evaluator import MetricsEvaluator

SEPARATOR = "=" * 65
THIN_SEP  = "─" * 65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config.yaml from the project root."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_models() -> list:
    """Instantiate all five baseline models."""
    return [
        NaiveForecaster(),
        MovingAverageForecaster(),
        SeasonalNaiveForecaster(),
        ARIMAForecaster(),
        ProphetForecaster(),
    ]


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """SupplyBench — Supply Chain Forecasting Benchmark."""
    pass


@cli.command()
@click.option(
    "--item",
    default=None,
    type=str,
    help="Benchmark a single item_id instead of all items.",
)
@click.option(
    "--cv",
    is_flag=True,
    default=False,
    help="Also run Rolling Origin Cross Validation after the benchmark.",
)
def run(item: str | None, cv: bool) -> None:
    """Run the full benchmark pipeline end-to-end."""

    try:
        # -------------------------------------------------------------- #
        # Step 1 — Welcome banner                                         #
        # -------------------------------------------------------------- #
        print(f"\n{SEPARATOR}")
        print("  SupplyBench — Supply Chain Forecasting Benchmark")
        print(SEPARATOR)

        # -------------------------------------------------------------- #
        # Step 2 — Load config                                             #
        # -------------------------------------------------------------- #
        print(f"\n{THIN_SEP}")
        print("  Configuration")
        print(THIN_SEP)

        config = _load_config()
        cfg_data = config["dataset"]
        cfg_eval = config["evaluation"]

        dataset_name  = cfg_data["name"]
        store_filter  = cfg_data["store_filter"]
        max_items     = int(cfg_data["max_items"])
        test_days     = int(cfg_eval["test_size"])
        output_path   = Path(config["report"]["output_path"])

        print(f"  Dataset      : {dataset_name}")
        print(f"  Store        : {store_filter}")
        print(f"  Max items    : {max_items}")
        print(f"  Test horizon : {test_days} days")
        print(f"  Output path  : {output_path}")

        # -------------------------------------------------------------- #
        # Step 3 — Load the M5 dataset                                     #
        # -------------------------------------------------------------- #
        print(f"\n{THIN_SEP}")
        print("  Loading Dataset")
        print(THIN_SEP)

        df = load_m5_data(config)

        print(f"  Shape      : {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Date range : {df['date'].min().date()} to {df['date'].max().date()}")

        # -------------------------------------------------------------- #
        # Step 4 — Determine which items to benchmark                      #
        # -------------------------------------------------------------- #
        all_items = df["item_id"].unique().tolist()

        if item is not None:
            if item not in all_items:
                print(f"\n  ERROR: item '{item}' not found in dataset.")
                print(f"  Available items ({len(all_items)}): {all_items[:5]} ...")
                sys.exit(1)
            items_to_run = [item]
            print(f"\n  Single-item mode: {item}")
        else:
            items_to_run = all_items[:max_items]
            print(f"\n  Running all {len(items_to_run)} items")

        # -------------------------------------------------------------- #
        # Step 5 — Run benchmark for each item                             #
        # -------------------------------------------------------------- #
        print(f"\n{THIN_SEP}")
        print("  Running Benchmark")
        print(THIN_SEP)

        evaluator = MetricsEvaluator()
        per_item_results: dict[str, dict] = {}
        skipped_items: list[str] = []

        for idx, item_id in enumerate(items_to_run, start=1):
            t0 = time.perf_counter()
            sys.stdout.write(f"  [{idx:>{len(str(len(items_to_run)))}}/{len(items_to_run)}]"
                             f"  {item_id:<30}")
            sys.stdout.flush()

            try:
                # Isolate this product's time series
                product_df = (
                    df[df["item_id"] == item_id]
                    .sort_values("date")
                    .reset_index(drop=True)
                )

                if len(product_df) <= test_days:
                    print(f"SKIPPED (only {len(product_df)} days, need >{test_days})")
                    skipped_items.append(item_id)
                    continue

                # Train / test split
                train_df = product_df.iloc[:-test_days]
                test_df  = product_df.iloc[-test_days:]

                train_series = train_df.set_index("date")["sales"]
                actual       = test_df["sales"].values.astype(np.float64)

                # Fit all models and collect predictions
                models = _build_models()
                predictions: dict[str, np.ndarray] = {}

                for model in models:
                    try:
                        model.fit(train_series)
                        preds = model.predict(test_days)
                        predictions[model.name] = preds
                    except Exception as e:
                        # Individual model failure — skip this model for this item
                        print(f"\n  [Warning] Model {model.name} failed: {e}")

                if not predictions:
                    elapsed = time.perf_counter() - t0
                    print(f"SKIPPED — all models failed ({elapsed:.2f}s)")
                    skipped_items.append(item_id)
                    continue

                # Evaluate
                item_metrics = evaluator.evaluate_all(actual, predictions)
                per_item_results[item_id] = item_metrics

                elapsed = time.perf_counter() - t0
                print(f"done ({elapsed:.2f}s)")

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"ERROR ({elapsed:.2f}s) — {exc}")
                skipped_items.append(item_id)
                continue

        # -------------------------------------------------------------- #
        # Step 6 — Compute aggregate results                               #
        # -------------------------------------------------------------- #
        if not per_item_results:
            print(f"\n  ERROR: No items were successfully evaluated.")
            sys.exit(1)

        # Collect per-model metric lists across all items
        model_names = set()
        for item_metrics in per_item_results.values():
            model_names.update(item_metrics.keys())
        model_names = sorted(model_names)

        aggregate: dict[str, dict] = {}

        for model_name in model_names:
            rmse_vals = []
            mae_vals  = []
            mape_vals = []

            for item_metrics in per_item_results.values():
                if model_name in item_metrics:
                    m = item_metrics[model_name]
                    rmse_vals.append(m["rmse"])
                    mae_vals.append(m["mae"])
                    if m["mape"] is not None:
                        mape_vals.append(m["mape"])

            aggregate[model_name] = {
                "avg_rmse": round(float(np.mean(rmse_vals)), 4) if rmse_vals else None,
                "avg_mae":  round(float(np.mean(mae_vals)), 4)  if mae_vals  else None,
                "avg_mape": round(float(np.mean(mape_vals)), 4) if mape_vals else None,
            }

        # Sort by avg_rmse (ascending — best first)
        ranked_aggregate = OrderedDict(
            sorted(aggregate.items(), key=lambda kv: kv[1]["avg_rmse"] or float("inf"))
        )

        # -------------------------------------------------------------- #
        # Step 7 — Print results table                                     #
        # -------------------------------------------------------------- #
        n_evaluated = len(per_item_results)

        print(f"\n{SEPARATOR}")
        print(f"  Benchmark Results — {n_evaluated} items — "
              f"{dataset_name.upper()} Dataset ({store_filter})")
        print(SEPARATOR)

        col_w = 14
        header = f"  {'Model':<20}{'Avg RMSE':>{col_w}}{'Avg MAE':>{col_w}}{'Avg MAPE(%)':>{col_w}}"
        print(f"\n{header}")
        print("  " + "─" * (20 + col_w * 3))

        for name, agg in ranked_aggregate.items():
            rmse_str = f"{agg['avg_rmse']:.4f}" if agg["avg_rmse"] is not None else "N/A"
            mae_str  = f"{agg['avg_mae']:.4f}"  if agg["avg_mae"]  is not None else "N/A"
            mape_str = f"{agg['avg_mape']:.2f}"  if agg["avg_mape"] is not None else "N/A"
            print(f"  {name:<20}{rmse_str:>{col_w}}{mae_str:>{col_w}}{mape_str:>{col_w}}")

        # -------------------------------------------------------------- #
        # Step 8 — Ranking summary                                         #
        # -------------------------------------------------------------- #
        print(f"\n{THIN_SEP}")
        print("  Model Ranking (by Avg RMSE, lower is better)")
        print(THIN_SEP)

        ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
        for rank, (name, agg) in enumerate(ranked_aggregate.items(), start=1):
            label = ordinals.get(rank, f"{rank}th")
            rmse_str = f"{agg['avg_rmse']:.4f}" if agg["avg_rmse"] is not None else "N/A"
            print(f"  {label:>4}  {name:<20} Avg RMSE = {rmse_str}")

        # Winner line
        winner_name = next(iter(ranked_aggregate))
        winner_rmse = ranked_aggregate[winner_name]["avg_rmse"]
        winner_str  = f"{winner_rmse:.4f}" if winner_rmse is not None else "N/A"

        print(f"\n{SEPARATOR}")
        print(f"  Winner: {winner_name} with Avg RMSE of {winner_str}")
        print(SEPARATOR)

        # -------------------------------------------------------------- #
        # Step 9 — Save results to JSON                                    #
        # -------------------------------------------------------------- #
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / "results.json"

        results_payload = {
            "dataset": dataset_name,
            "store": store_filter,
            "items_evaluated": n_evaluated,
            "test_days": test_days,
            "aggregate_results": dict(ranked_aggregate),
            "per_item_results": per_item_results,
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2, ensure_ascii=False)

        # -------------------------------------------------------------- #
        # Step 10 — AI Explanation via Groq                                #
        # -------------------------------------------------------------- #
        print(f"\n  Results saved to: {results_file}")

        print("\n  Generating AI explanation via Groq...")
        from explainer import GroqExplainer

        groq_explainer = GroqExplainer()
        explanation = groq_explainer.explain_from_file(str(results_file))

        print(f"\n{THIN_SEP}")
        print("  AI Explanation")
        print(THIN_SEP)
        print(explanation)
        print(THIN_SEP)

        # Save the explanation back into results.json
        results_payload["ai_explanation"] = explanation
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2, ensure_ascii=False)

        # -------------------------------------------------------------- #
        # Step 11 — Completion message                                     #
        # -------------------------------------------------------------- #
        if skipped_items:
            print(f"\n  Note: {len(skipped_items)} item(s) were skipped due to errors.")

        # -------------------------------------------------------------- #
        # Step 12 — Rolling Origin Cross Validation (optional)             #
        # -------------------------------------------------------------- #
        if cv:
            from evaluator import RollingOriginCV

            cv_n_folds = 5
            cv_test_size = test_days

            print(f"\n{SEPARATOR}")
            print(f"  Rolling Origin Cross Validation "
                  f"({cv_n_folds} folds × {cv_test_size} days)")
            print(SEPARATOR)

            cv_engine = RollingOriginCV(
                n_folds=cv_n_folds,
                test_size=cv_test_size,
            )

            cv_models = _build_models()
            cv_results = cv_engine.evaluate_all_items(
                df, cv_models, show_progress=True
            )

            # Print CV results table
            print(f"\n{THIN_SEP}")
            print("  Cross Validation Results (mean ± std across all folds & items)")
            print(THIN_SEP)

            col_w = 14
            cv_header = (f"  {'Model':<20}"
                         f"{'Mean RMSE':>{col_w}}"
                         f"{'Std RMSE':>{col_w}}"
                         f"{'Mean MAPE':>{col_w}}"
                         f"{'Std MAPE':>{col_w}}")
            print(f"\n{cv_header}")
            print("  " + "─" * (20 + col_w * 4))

            # Sort by mean_rmse
            cv_ranked = sorted(
                cv_results["models"].items(),
                key=lambda kv: kv[1]["mean_rmse"] or float("inf"),
            )

            for name, m in cv_ranked:
                rmse_s = f"{m['mean_rmse']:.4f}" if m["mean_rmse"] is not None else "N/A"
                rmse_std = f"±{m['std_rmse']:.4f}" if m["std_rmse"] is not None else "N/A"
                mape_s = f"{m['mean_mape']:.2f}%" if m["mean_mape"] is not None else "N/A"
                mape_std = f"±{m['std_mape']:.2f}%" if m["std_mape"] is not None else "N/A"
                print(f"  {name:<20}{rmse_s:>{col_w}}{rmse_std:>{col_w}}"
                      f"{mape_s:>{col_w}}{mape_std:>{col_w}}")

            # Save CV results to JSON
            cv_results_file = output_path / "cv_results.json"
            with open(cv_results_file, "w", encoding="utf-8") as f:
                json.dump(cv_results, f, indent=2, ensure_ascii=False)
            print(f"\n  CV results saved to: {cv_results_file}")

        # -------------------------------------------------------------- #
        # Step 13 — Generate HTML report                                   #
        # -------------------------------------------------------------- #
        print("\n  Generating HTML report...")
        from report import HTMLReportGenerator

        report_gen = HTMLReportGenerator()
        cv_results_path = output_path / "cv_results.json"
        report_gen.generate_from_file(
            str(results_file),
            str(output_path / "report.html"),
            cv_results_path=str(cv_results_path) if cv_results_path.exists() else None,
        )
        print("  Open reports/report.html in your browser to view the full report.")

        print(f"\n{SEPARATOR}")
        print("  SupplyBench run complete.")
        print(f"{SEPARATOR}\n")

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
