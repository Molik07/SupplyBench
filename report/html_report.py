"""
html_report.py — HTML Report Generator
========================================
Reads reports/results.json and optionally reports/cv_results.json,
then generates a clean, professional HTML report card using Jinja2
templates and Plotly charts.
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

    def generate(
        self,
        results: dict,
        output_path: str,
        cv_results: dict | None = None,
    ) -> None:
        """Generate the HTML report from a results dictionary.

        Parameters
        ----------
        results : dict
            The full results dictionary loaded from results.json.
        output_path : str
            File path where the HTML report will be saved.
        cv_results : dict, optional
            Rolling Origin CV results loaded from cv_results.json.
            If None, the CV sections are omitted from the report.
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

        # ── Step 3: Build CV template variables (if CV data exists) ──── #
        has_cv = cv_results is not None and "models" in (cv_results or {})

        cv_template_vars = {}
        if has_cv:
            cv_template_vars = self._build_cv_template_vars(
                cv_results, winner_name
            )

        # ── Step 4: Load and render Jinja2 template ──────────────────── #
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
            has_cv=has_cv,
            **cv_template_vars,
        )

        # ── Step 5: Write the output file ────────────────────────────── #
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # ── Step 6: Confirmation ─────────────────────────────────────── #
        print(f"[SupplyBench] HTML report saved to {output_path}")

    def generate_from_file(
        self,
        results_json_path: str,
        output_path: str,
        cv_results_path: str | None = None,
    ) -> None:
        """Load results from JSON file(s) and generate the HTML report.

        Parameters
        ----------
        results_json_path : str
            Path to results.json.
        output_path : str
            File path where the HTML report will be saved.
        cv_results_path : str, optional
            Path to cv_results.json. If None, CV sections are omitted.
        """
        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        cv_results = None
        if cv_results_path and os.path.exists(cv_results_path):
            with open(cv_results_path, "r", encoding="utf-8") as f:
                cv_results = json.load(f)

        self.generate(results, output_path, cv_results=cv_results)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_plotly_chart(ranked_models: list, winner_name: str) -> str:
        """Build a Plotly bar chart comparing models by Avg RMSE."""

        names = [m["name"] for m in ranked_models]
        rmse_values = [m["avg_rmse"] or 0 for m in ranked_models]

        colors = [
            "#c4a882" if name == winner_name else "#3a3530"
            for name in names
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names,
            y=rmse_values,
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="%{x}<br>Avg RMSE: %{y:.4f}<extra></extra>",
        ))

        fig.update_layout(
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(family="Inter, sans-serif", color="#8a8680", size=11),
            title=dict(
                text="AVG RMSE BY MODEL",
                font=dict(size=11, color="#8a8680"),
                x=0,
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(color="#8a8680", size=11),
                linecolor="#2a2a2a",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#2a2a2a",
                zeroline=False,
                tickfont=dict(color="#8a8680", size=11),
                linecolor="#2a2a2a",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=280,
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def _build_cv_chart(cv_ranked_models: list, cv_winner_name: str) -> str:
        """Build a grouped bar chart showing Mean RMSE and Mean RMSE + Std."""

        names = [m["name"] for m in cv_ranked_models]
        mean_values = [m["mean_rmse"] or 0 for m in cv_ranked_models]
        upper_values = [
            (m["mean_rmse"] or 0) + (m["std_rmse"] or 0)
            for m in cv_ranked_models
        ]

        colors_mean = [
            "#c4a882" if name == cv_winner_name else "#3a3530"
            for name in names
        ]
        colors_upper = [
            "#8a7a62" if name == cv_winner_name else "#2a2520"
            for name in names
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Mean RMSE",
            x=names,
            y=mean_values,
            marker_color=colors_mean,
            marker_line_width=0,
            hovertemplate="%{x}<br>Mean RMSE: %{y:.4f}<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="Mean RMSE + 1 Std",
            x=names,
            y=upper_values,
            marker_color=colors_upper,
            marker_line_width=0,
            hovertemplate="%{x}<br>Mean + Std: %{y:.4f}<extra></extra>",
        ))

        fig.update_layout(
            barmode="group",
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(family="Inter, sans-serif", color="#8a8680", size=11),
            title=dict(
                text="CROSS VALIDATION — MEAN RMSE vs MEAN + STD",
                font=dict(size=11, color="#8a8680"),
                x=0,
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(color="#8a8680", size=11),
                linecolor="#2a2a2a",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#2a2a2a",
                zeroline=False,
                tickfont=dict(color="#8a8680", size=11),
                linecolor="#2a2a2a",
            ),
            legend=dict(
                font=dict(color="#8a8680", size=10),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=300,
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    def _build_cv_template_vars(
        self,
        cv_results: dict,
        standard_winner_name: str,
    ) -> dict:
        """Build all template variables for the CV sections."""

        cv_models = cv_results.get("models", {})
        cv_n_folds = cv_results.get("n_folds", 5)
        cv_test_size = cv_results.get("test_size", 28)

        # Sort by mean_rmse ascending
        cv_ranked = sorted(
            cv_models.items(),
            key=lambda kv: kv[1].get("mean_rmse") or float("inf"),
        )

        # Build ranked model list with consistency ratings
        cv_ranked_models = []
        for name, metrics in cv_ranked:
            std_rmse = metrics.get("std_rmse")

            if std_rmse is not None:
                if std_rmse < 0.15:
                    consistency = "High"
                elif std_rmse < 0.30:
                    consistency = "Moderate"
                else:
                    consistency = "Variable"
            else:
                consistency = "Variable"

            cv_ranked_models.append({
                "name": name,
                "mean_rmse": metrics.get("mean_rmse"),
                "std_rmse": metrics.get("std_rmse"),
                "mean_mae": metrics.get("mean_mae"),
                "std_mae": metrics.get("std_mae"),
                "mean_mape": metrics.get("mean_mape"),
                "std_mape": metrics.get("std_mape"),
                "consistency_rating": consistency,
            })

        # CV winner
        cv_winner_name = cv_ranked_models[0]["name"] if cv_ranked_models else "N/A"
        cv_winner_mean_rmse = cv_ranked_models[0]["mean_rmse"] if cv_ranked_models else None
        cv_winner_std_rmse = cv_ranked_models[0]["std_rmse"] if cv_ranked_models else None

        # Generate insight text
        if cv_winner_name != standard_winner_name:
            cv_insight_text = (
                f"Interesting finding: {standard_winner_name} ranks first on a single "
                f"evaluation window, but {cv_winner_name} demonstrates more consistent "
                f"performance across {cv_n_folds} time windows with a lower average RMSE of "
                f"{cv_winner_mean_rmse:.4f} ± {cv_winner_std_rmse:.4f}. "
                f"In production environments where consistency matters as much as peak "
                f"accuracy, {cv_winner_name} may be the more reliable choice."
            )
        else:
            cv_insight_text = (
                f"{cv_winner_name} ranks first on both single evaluation and cross validation, "
                f"confirming it as the most reliable model for this dataset with consistent "
                f"performance across {cv_n_folds} time windows."
            )

        # Build CV chart
        cv_plotly_chart_html = self._build_cv_chart(cv_ranked_models, cv_winner_name)

        return {
            "cv_n_folds": cv_n_folds,
            "cv_test_size": cv_test_size,
            "cv_winner_name": cv_winner_name,
            "cv_winner_mean_rmse": cv_winner_mean_rmse,
            "cv_winner_std_rmse": cv_winner_std_rmse,
            "cv_ranked_models": cv_ranked_models,
            "cv_plotly_chart_html": cv_plotly_chart_html,
            "cv_insight_text": cv_insight_text,
            "standard_winner_name": standard_winner_name,
        }
