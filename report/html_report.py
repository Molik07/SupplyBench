"""
html_report.py — HTML Report Generator
========================================
Reads reports/results.json and generates a clean, professional
HTML report card using Jinja2 templates and Plotly charts.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader


class HTMLReportGenerator:
    """Generates a self-contained HTML benchmark report."""

    def __init__(self):
        # Templates live in report/templates/ next to this file
        template_dir = Path(__file__).resolve().parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, results: dict, output_path: str) -> None:
        """Generate the HTML report from a results dictionary.

        Parameters
        ----------
        results : dict
            The full results dictionary loaded from results.json.
        output_path : str
            File path where the HTML report will be saved.
        """

        # ── Step 1: Extract data ─────────────────────────────────────── #
        dataset = results.get("dataset", "unknown")
        store = results.get("store", "unknown")
        items_evaluated = results.get("items_evaluated", 0)
        test_days = results.get("test_days", 0)
        aggregate = results.get("aggregate_results", {})
        ai_explanation = results.get("ai_explanation", "No AI explanation available.")

        # Sort models by avg_rmse ascending (best first)
        ranked_models = []
        for name, metrics in sorted(
            aggregate.items(),
            key=lambda kv: kv[1].get("avg_rmse") or float("inf"),
        ):
            ranked_models.append({
                "name": name,
                "avg_rmse": metrics.get("avg_rmse"),
                "avg_mae": metrics.get("avg_mae"),
                "avg_mape": metrics.get("avg_mape"),
            })

        winner_name = ranked_models[0]["name"] if ranked_models else "N/A"
        winner_rmse = ranked_models[0]["avg_rmse"] if ranked_models else None

        # ── Step 2: Generate Plotly bar chart ────────────────────────── #
        plotly_chart_html = self._build_plotly_chart(ranked_models, winner_name)

        # ── Step 3: Load and render Jinja2 template ──────────────────── #
        generated_at = datetime.now().strftime("%B %d, %Y at %H:%M")

        template = self.env.get_template("report.html")
        html_content = template.render(
            dataset=dataset,
            store=store,
            items_evaluated=items_evaluated,
            test_days=test_days,
            winner_name=winner_name,
            winner_rmse=winner_rmse,
            ranked_models=ranked_models,
            plotly_chart_html=plotly_chart_html,
            ai_explanation=ai_explanation,
            generated_at=generated_at,
        )

        # ── Step 4: Write the output file ────────────────────────────── #
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # ── Step 5: Confirmation ─────────────────────────────────────── #
        print(f"[SupplyBench] HTML report saved to {output_path}")

    def generate_from_file(self, results_json_path: str, output_path: str) -> None:
        """Load results from a JSON file and generate the HTML report.

        Parameters
        ----------
        results_json_path : str
            Path to results.json.
        output_path : str
            File path where the HTML report will be saved.
        """
        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        self.generate(results, output_path)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_plotly_chart(ranked_models: list, winner_name: str) -> str:
        """Build a Plotly bar chart comparing models by Avg RMSE."""

        names = [m["name"] for m in ranked_models]
        rmse_values = [m["avg_rmse"] or 0 for m in ranked_models]

        # Color the winner green, others a neutral blue
        colors = [
            "#10b981" if name == winner_name else "#6366f1"
            for name in names
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=names,
                    y=rmse_values,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in rmse_values],
                    textposition="outside",
                    hovertemplate="%{x}<br>Avg RMSE: %{y:.4f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title={
                "text": "Model Performance Comparison — Average RMSE",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18, "color": "#1e293b"},
            },
            xaxis_title="Model",
            yaxis_title="Average RMSE",
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            font={"family": "Inter, system-ui, sans-serif", "color": "#334155"},
            margin={"t": 60, "b": 60, "l": 60, "r": 30},
            yaxis={"gridcolor": "#e2e8f0"},
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
