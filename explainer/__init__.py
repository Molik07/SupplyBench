"""
explainer — AI Explanation Layer for SupplyBench
=================================================
Uses the Groq API to generate plain-English interpretations
of benchmark results.
"""

from .groq_explainer import GroqExplainer

__all__ = ["GroqExplainer"]
