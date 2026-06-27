# SupplyBench

SupplyBench is a comprehensive supply chain forecasting benchmark tool. It tests different baseline models (Naive, Moving Average, Seasonal Naive, ARIMA) on the M5 dataset, generates plain-English insights using Groq AI (Llama models), and creates a beautiful, professional HTML report.

## Features
- **Data Loader**: Preprocesses M5 dataset for forecasting.
- **Baseline Models**: Includes `NaiveForecaster`, `MovingAverageForecaster`, `SeasonalNaiveForecaster`, and `ARIMAForecaster`.
- **Metrics Evaluator**: Evaluates predictions using RMSE, MAE, and MAPE.
- **Groq Explainer (AI Insights)**: Leverages Groq's API and the `llama-3.3-70b-versatile` model to provide clear, actionable business insights from the benchmark metrics.
- **HTML Report Generator**: Produces an interactive, self-contained HTML report with Plotly charts and highlighted metrics.
- **CLI**: A single command-line interface to run the entire pipeline end-to-end.

## Installation

1. Clone the repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Groq API key:
   Create a `.env` file in the root directory and add your key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

Run the entire end-to-end benchmark on all products (configured in `config.yaml`):

```bash
python cli.py run
```

Run the benchmark for a single specific item:

```bash
python cli.py run --item HOBBIES_1_001
```

Once the run is complete, open `reports/report.html` in any web browser to view the generated benchmarking report.

## Tests
You can verify individual components using the included test scripts:
- `test_loader.py`
- `test_metrics.py`
- `test_baselines.py`
- `test_explainer.py`
- `test_report.py`

## Configuration
Modify `config.yaml` to change the dataset filter, evaluation horizon, Groq model, and output paths.
