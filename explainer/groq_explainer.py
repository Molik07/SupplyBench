"""
groq_explainer.py — Groq AI Explanation Layer
==============================================
Reads benchmark results and sends them to the Groq API to generate
a plain-English interpretation of what the numbers mean.

This is what makes SupplyBench more than just a metrics calculator —
it tells you what the results actually mean in plain language that
both engineers and business stakeholders can understand.
"""

import json
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a supply chain analytics expert who specializes in demand forecasting. "
    "Your job is to interpret benchmarking results and explain them clearly to both "
    "technical and non-technical stakeholders. Be specific, reference the actual "
    "numbers from the results, and provide actionable recommendations. "
    "Keep your response concise but insightful — around 150 to 200 words."
)


class GroqExplainer:
    """Generates plain-English explanations of benchmark results via Groq."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def explain(self, results: dict) -> str:
        """Take a full results dictionary and return an AI explanation string.

        Parameters
        ----------
        results : dict
            The full results dictionary as saved in results.json.

        Returns
        -------
        str
            Plain-English explanation of the benchmark results.
        """

        # ── Step 1: Extract key information ──────────────────────────── #
        dataset = results.get("dataset", "unknown")
        store = results.get("store", "unknown")
        items_evaluated = results.get("items_evaluated", 0)
        test_days = results.get("test_days", 0)
        aggregate = results.get("aggregate_results", {})

        # ── Step 2: Build the user message ───────────────────────────── #
        user_message = self._format_user_message(
            dataset, store, items_evaluated, test_days, aggregate
        )

        # ── Step 3: Call the Groq API ────────────────────────────────── #
        try:
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY")
            client = Groq(api_key=api_key)

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # ── Step 4: Return the response text ─────────────────────── #
            response_text = completion.choices[0].message.content
            return response_text

        except Exception as exc:
            # ── Step 5: Graceful fallback on failure ──────────────────── #
            print(f"[SupplyBench] Groq API call failed: {exc}")
            return (
                "AI explanation unavailable. Please check your "
                "GROQ_API_KEY environment variable and try again."
            )

    def explain_from_file(self, file_path: str) -> str:
        """Load results from a JSON file and return an AI explanation.

        Parameters
        ----------
        file_path : str
            Path to a results.json file.

        Returns
        -------
        str
            Plain-English explanation of the benchmark results.
        """
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        return self.explain(results)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_user_message(
        dataset: str,
        store: str,
        items_evaluated: int,
        test_days: int,
        aggregate: dict,
    ) -> str:
        """Format the benchmark results into a structured prompt."""

        # Sort models by avg_rmse (ascending) to rank them
        ranked = sorted(
            aggregate.items(),
            key=lambda kv: kv[1].get("avg_rmse") or float("inf"),
        )

        # Build the numbered model lines
        model_lines = []
        for rank, (name, metrics) in enumerate(ranked, start=1):
            rmse = metrics.get("avg_rmse")
            mae = metrics.get("avg_mae")
            mape = metrics.get("avg_mape")

            rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
            mae_str = f"{mae:.4f}" if mae is not None else "N/A"
            mape_str = f"{mape:.2f}" if mape is not None else "N/A"

            model_lines.append(
                f"{rank}. {name}: Avg RMSE={rmse_str}, "
                f"Avg MAE={mae_str}, Avg MAPE={mape_str}%"
            )

        models_block = "\n".join(model_lines)

        return (
            f"Here are the SupplyBench benchmark results for {dataset} dataset, "
            f"store {store}, evaluated across {items_evaluated} products over a "
            f"{test_days}-day test horizon.\n\n"
            f"Model Performance (ranked by Average RMSE, lower is better):\n\n"
            f"{models_block}\n\n"
            f"Please provide:\n"
            f"1. A clear interpretation of these results\n"
            f"2. Which model performed best and why it likely outperformed the others\n"
            f"3. What the MAPE values tell us about forecast accuracy in practical terms\n"
            f"4. One specific recommendation for a supply chain team using these results"
        )
