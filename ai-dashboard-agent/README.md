# AI Dashboard Agent

An AI-powered Streamlit app that turns any CSV or Excel file into an interactive, insight-driven dashboard with no manual chart configuration needed.

Upload a file, optionally describe what you want to explore, and the app profiles your data, asks an LLM to plan the best charts, validates the plan, and renders a live Plotly dashboard with sidebar filters.

## Architecture

The app follows a four-stage pipeline:

```
Upload -> Ingest -> Profile/EDA -> LLM Plan -> Validate -> Render
```

| Stage | What happens |
|---|---|
| **Ingest** | Reads CSV/Excel. Drops empty rows/columns, sanitises column names, auto-detects dates and numeric strings. Results are cached by file content so re-uploads are instant. |
| **Profile + EDA** | Each column gets a semantic role (dimension, measure, temporal, year, high-cardinality dimension, identifier, free-text). Optional statistical EDA computes skewness, correlations, and near-zero variance on a 5K-row sample. EDA results are cached by dataset signature. |
| **Plan** | A schema summary (JSON) is sent to Llama 3.3 70B via Groq. The LLM returns a Pydantic-validated DashboardPlan with 3-5 ChartSpec objects and filter columns. A detailed system prompt encodes chart-selection rules, aggregation guidance, color caps, and diversity constraints. Includes retry logic with exponential backoff (up to 3 attempts). |
| **Validate and Render** | Every chart spec is checked against the real DataFrame: fuzzy column-name repair, type-specific guardrails, and color cardinality caps. Invalid charts are silently dropped. Surviving charts are aggregated using efficient pandas `.agg()` and rendered with theme-aware Plotly. |

### Key design decisions

- **Structured output over free-form JSON.** Pydantic schemas with `with_structured_output()` eliminate JSON parsing bugs.
- **Role-based column profiling.** The LLM never sees raw data, only a typed schema summary. This keeps the prompt small and focused.
- **Two-pass safety.** The LLM proposes, then `validate_and_repair_chart()` enforces hard constraints. Fuzzy matching fixes hallucinated column names.
- **Theme-aware rendering.** Plotly charts automatically match Streamlit's light or dark theme instead of forcing a hardcoded style.
- **Resilient LLM calls.** Retry logic with exponential backoff handles rate limits, timeouts, and transient failures gracefully.

## Features

- **7 chart types:** bar, line, area, histogram, scatter, pie (donut), box
- **Automatic aggregation:** sum, mean, or count chosen by the LLM based on data shape
- **Smart column roles:** year detection, high-cardinality dimensions (26-100 unique values kept as filters), person-name heuristics
- **Intent-driven planning:** free-text input steers the LLM toward charts answering your question
- **Interactive sidebar filters:** numeric sliders, searchable multiselect for categoricals of any size
- **Chart type switcher:** override the LLM's choice per panel without regenerating the plan
- **Auto log-scale:** applied when value range exceeds 100x on line/scatter/area
- **Time-series coarsening:** daily timestamps auto-floored to monthly when >52 unique dates
- **Top-N limiting:** bar, area, and pie charts capped at top 15 categories by default
- **Data quality panel:** null percentages, role assignments, sample values at a glance
- **EDA panel:** skewness flags, correlation pairs (r > 0.6), near-zero variance warnings
- **PNG download:** one-click export per chart with a clear message if kaleido is not installed
- **Caching:** file ingestion and EDA are cached with `st.cache_data` so rerenders and re-uploads are fast
- **Retry logic:** LLM calls retry up to 3 times with 1s/3s/7s backoff on failure
- **Theme-aware charts:** Plotly template adapts to Streamlit's light or dark mode automatically

## Quickstart

```bash
pip install streamlit pandas plotly python-dotenv langchain-groq langchain-core pydantic openpyxl

# Optional but recommended for PNG chart exports
pip install -U kaleido

echo "GROQ_API_KEY=gsk_..." > .env

streamlit run app.py
```

## Use Cases

- **Sales teams:** upload a CRM export, see revenue by region, rep performance, deal-stage distribution
- **Marketing analysts:** campaign spend vs. conversion scatter, channel breakdowns, time trends
- **Product managers:** user-event logs, feature adoption over time, engagement distributions
- **Data journalists:** quick visual EDA on public datasets before writing a story
- **Ad-hoc analysis:** skip the 30 minutes of chart-building in Excel

## Optimisations

| Technique | Why |
|---|---|
| `st.cache_data` on ingestion and EDA | Eliminates redundant parsing and profiling across Streamlit rerenders |
| 5K-row sampling for EDA | Fast profiling on large files without losing statistical signal |
| Schema-only LLM context | Column metadata only, not raw rows: minimal tokens, faster inference |
| Groq (Llama 3.3 70B) at temp 0.1 | Sub-second structured generation; low temperature for deterministic plans |
| `.groupby().agg()` over `.apply(lambda)` | 2-3x faster aggregation, no pandas deprecation warnings |
| Pydantic model_validator | Hard cap at 5 charts before any rendering work |
| Fuzzy column repair | Avoids a full LLM retry when the model slightly misspells a column |
| Monthly flooring for dense time series | Prevents unreadable daily-granularity trend charts |
| Retry with exponential backoff | Handles Groq rate limits and transient errors without user intervention |

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| Charts | Plotly Express |
| LLM | Llama 3.3 70B via Groq |
| Orchestration | LangChain Core |
| Validation | Pydantic v2 |
| Data | pandas, openpyxl |

## License

MIT

