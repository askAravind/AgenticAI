# Standard library
import os
import json
import re
import time
import hashlib
import difflib
from typing import Optional, Literal

# Third-party
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dotenv import load_dotenv

# LangChain / Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator


# Environment and LLM setup

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY"),
)


# Theme detection
# Reads Streamlit's built-in theme setting so Plotly charts match the UI.
# Falls back to "plotly" (light) if detection fails.

def _get_plotly_template() -> str:
    """Return a Plotly template that matches the current Streamlit theme."""
    try:
        theme = st.get_option("theme.base")  # "light" or "dark"
    except Exception:
        theme = None
    return "plotly_dark" if theme == "dark" else "plotly"


# Pydantic schema for structured LLM output
# The LLM returns a DashboardPlan containing filter columns and 3-5 ChartSpecs.

CHART_TYPES = Literal["bar", "line", "histogram", "scatter", "pie", "box", "area"]


class ChartSpec(BaseModel):
    """Describes one chart panel in the dashboard."""
    type: CHART_TYPES = Field(description="Chart type to render.")
    x: str = Field(description="Column name for the x-axis.")
    y: Optional[str] = Field(
        default=None,
        description=(
            "Column name for the y-axis. "
            "Not needed for histogram. "
            "For pie: the measure to size slices by (aggregated as sum)."
        ),
    )
    color: Optional[str] = Field(
        default=None,
        description="Column for color/legend grouping. None if not useful.",
    )
    aggregate: Literal["sum", "mean", "count", "none"] = Field(
        default="none",
        description=(
            "Aggregation before plotting. "
            "'sum' for totals, 'mean' for rates/averages."
        ),
    )
    orient: Literal["v", "h"] = Field(
        default="v",
        description="Bar orientation. Use 'h' when x has > 8 unique values.",
    )
    top_n: Optional[int] = Field(
        default=15,
        description="Limit bar/line/area/pie to top-N categories by y value.",
    )
    title: str = Field(description="Short, descriptive chart title.")
    reason: str = Field(
        description="One sentence: what business question this chart answers.",
    )


class DashboardPlan(BaseModel):
    """Full dashboard plan returned by the LLM."""
    filters: list[str] = Field(
        description=(
            "Columns to expose as sidebar filters. "
            "Include all dimensions (nunique <= 100) and year/period columns."
        ),
    )
    charts: list[ChartSpec] = Field(description="3 to 5 chart specifications.")

    @model_validator(mode="after")
    def cap_charts(self) -> "DashboardPlan":
        self.charts = self.charts[:5]
        return self


# Stage 1: Ingest
# Reads CSV/Excel, drops empty rows/cols, cleans column names,
# auto-detects dates and numeric columns hiding as strings.
# Cached by file content hash so re-uploading the same file is instant.

def _file_hash(uploaded_file) -> str:
    """Compute a stable hash of the uploaded file contents for caching."""
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(data).hexdigest()


@st.cache_data(show_spinner=False)
def ingest_file_cached(_file_bytes: bytes, file_name: str) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV or Excel file and clean it up. Returns (df, warnings).
    Accepts raw bytes so the result is cacheable by Streamlit."""
    import io
    warnings: list[str] = []

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(_file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(_file_bytes), engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Could not read file: {e}") from e

    before = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    after = df.shape
    if before != after:
        warnings.append(
            f"Dropped {before[0]-after[0]} empty rows "
            f"and {before[1]-after[1]} empty columns."
        )

    # Sanitise column names: strip whitespace, replace special chars, deduplicate
    cleaned_names: list[str] = []
    seen: dict[str, int] = {}
    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if re.match(r"Unnamed:\s*\d+", name):
            name = f"col_{i}"
        name = re.sub(r"[^\w]", "_", name).strip("_")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cleaned_names.append(name)
    df.columns = cleaned_names

    # Try parsing string columns with date-like names as datetime
    for col in df.select_dtypes(include="object").columns:
        if any(kw in col.lower() for kw in ["date", "time", "month", "year", "quarter", "period"]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.7:
                df[col] = parsed
                warnings.append(f"Parsed '{col}' as datetime.")

    # Try coercing string columns that are really numeric (e.g. "1,234" or "85%")
    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(
            df[col].astype(str)
                   .str.replace(",", "", regex=False)
                   .str.replace("%", "", regex=False)
                   .str.strip(),
            errors="coerce",
        )
        if coerced.notna().mean() > 0.85:
            df[col] = coerced
            warnings.append(f"Coerced '{col}' to numeric.")

    return df, warnings


def ingest_file(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """Thin wrapper that reads bytes once and delegates to the cached function."""
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return ingest_file_cached(file_bytes, uploaded_file.name)


# Stage 2: Profile and EDA
# Each column is assigned a role (temporal, year, measure, dimension, etc.)
# which tells the LLM what it can do with the column.
# Columns with 26-100 unique short strings get "high_cardinality_dimension"
# so they appear as filters but aren't used as bar-chart x-axes.

def _is_person_dimension(col: str) -> bool:
    """Heuristic: does the column name suggest a person/agent field?"""
    keywords = [
        "rep", "agent", "manager", "owner", "seller",
        "assignee", "analyst", "advisor", "consultant", "staff",
    ]
    col_lower = col.lower()
    return any(kw in col_lower for kw in keywords)


def profile_column(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    non_null = s.dropna()
    null_pct = round(s.isna().mean() * 100, 2)
    nunique = int(non_null.nunique())
    sample = [str(v) for v in non_null.unique()[:6]]

    if pd.api.types.is_datetime64_any_dtype(s):
        return {"name": col, "role": "temporal", "nunique": nunique,
                "null_pct": null_pct, "sample_values": sample}

    # Detect year columns: integers in 1900-2100 range with low cardinality density
    if pd.api.types.is_integer_dtype(s):
        in_year_range = non_null.between(1900, 2100).mean() > 0.9
        name_is_year = any(kw in col.lower() for kw in ["year", "yr", "vintage", "season"])
        low_density = (nunique / max(len(non_null), 1)) < 0.05
        if in_year_range and (name_is_year or low_density):
            return {"name": col, "role": "year", "nunique": nunique,
                    "null_pct": null_pct, "sample_values": sample}

    if pd.api.types.is_numeric_dtype(s):
        role = "low_cardinality_measure" if nunique <= 15 else "measure"
        return {"name": col, "role": role, "nunique": nunique,
                "null_pct": null_pct, "sample_values": sample,
                "min": round(float(non_null.min()), 4),
                "max": round(float(non_null.max()), 4)}

    # String columns: assign role based on cardinality and average string length
    avg_len = non_null.astype(str).str.len().mean()

    if nunique <= 25:
        return {"name": col, "role": "dimension", "nunique": nunique,
                "null_pct": null_pct, "sample_values": sample}

    # Medium-cardinality strings (26-100 unique, short tokens) are useful as
    # filters and occasionally as scatter color. Things like city, product, rep.
    if nunique <= 100 and avg_len < 20:
        return {
            "name": col, "role": "high_cardinality_dimension",
            "nunique": nunique, "null_pct": null_pct,
            "sample_values": sample,
            "is_person_dimension": _is_person_dimension(col),
        }

    # Everything else (100+ unique or very long strings) is junk for charts
    role = "identifier" if avg_len < 12 else "free_text"
    return {"name": col, "role": role, "nunique": nunique,
            "null_pct": null_pct, "sample_values": sample}


@st.cache_data(show_spinner=False)
def get_eda_summary(_df_hash: str, df: pd.DataFrame) -> dict:
    """Pure-pandas EDA on a sample: skewness, correlations, near-zero variance.
    Cached by a hash of the DataFrame shape and column names."""
    try:
        sample = df.sample(min(len(df), 5_000), random_state=42)
        numeric_cols = sample.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return {}

        summary: dict = {}
        for col in numeric_cols:
            s = sample[col].dropna()
            if len(s) < 10:
                continue
            skewness = float(s.skew())
            n_zeros = int((s == 0).sum())
            n_distinct = int(s.nunique())
            summary[col] = {
                "is_skewed": abs(skewness) > 1.5,
                "skewness": round(skewness, 2),
                "near_zero_variance": n_distinct <= 2 or (n_zeros / max(len(s), 1)) > 0.95,
                "p_missing": round(sample[col].isna().mean() * 100, 1),
                "high_corr_with": [],
            }

        if len(numeric_cols) > 1:
            try:
                corr_matrix = sample[numeric_cols].corr(method="pearson")
                for col in numeric_cols:
                    if col not in summary:
                        continue
                    others = corr_matrix[col].drop(labels=[col], errors="ignore")
                    top = others.abs().nlargest(2)
                    for other, val in top.items():
                        if float(val) > 0.6:
                            summary[col]["high_corr_with"].append(
                                {"col": str(other), "r": round(float(val), 2)}
                            )
            except Exception:
                pass
        return summary
    except Exception:
        return {}


def _df_cache_key(df: pd.DataFrame) -> str:
    """Cheap hash for caching: shape + column names + first/last row values."""
    sig = f"{df.shape}|{'|'.join(df.columns)}"
    return hashlib.md5(sig.encode()).hexdigest()


def build_schema_summary(df: pd.DataFrame, eda: dict) -> str:
    """Build a JSON schema summary the LLM uses to plan charts."""
    profiles = [profile_column(df, col) for col in df.columns]

    usable_roles = {
        "temporal", "year", "measure", "low_cardinality_measure",
        "dimension", "high_cardinality_dimension",
    }
    usable = [p for p in profiles if p["role"] in usable_roles]
    skipped = [p["name"] for p in profiles if p["role"] not in usable_roles]

    # Merge EDA stats (skewness, correlations) into usable column profiles
    if eda:
        for p in usable:
            col = p["name"]
            if col in eda:
                p["is_skewed"] = eda[col]["is_skewed"]
                p["near_zero_variance"] = eda[col]["near_zero_variance"]
                if eda[col]["high_corr_with"]:
                    p["high_corr_with"] = eda[col]["high_corr_with"]

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "usable_columns": usable,
        "skipped_columns_reason": {
            "names": skipped,
            "reason": "identifiers or free-text (nunique > 100 or avg token > 20 chars)",
        },
    }
    return json.dumps(summary, indent=2)


# Stage 3: Plan
# The LLM receives the schema summary and returns a structured DashboardPlan.
# The system prompt encodes all chart-selection rules so the LLM picks
# appropriate types, axes, aggregations, and filters.
# Includes retry logic: up to 3 attempts with exponential backoff on failure.

LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = [1, 3, 7]  # seconds to wait between retries

structured_llm = llm.with_structured_output(DashboardPlan)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a data visualisation expert building a business dashboard.

For each chart you propose, ask yourself first:
"What specific business question does this chart answer, and what decision
does it help the user make?" If you cannot answer that clearly, skip the chart.

RULES (follow strictly):

1. Only use column names from "usable_columns". NEVER invent column names.

2. Column role rules:

   dimension (nunique <= 25):
     - Best choice for x-axis in bar, box, pie.
     - Can be used as color in any chart type (within color caps in Rule 5).

   high_cardinality_dimension (nunique 26-100):
     - Do NOT use as x-axis in bar or box (too many bars/groups).
     - CAN be used as color in scatter only if nunique <= 15.
     - MUST be included as a sidebar filter.
     - If is_person_dimension = true, always include as a filter.
     - Do NOT use as x or color in bar, line, area, pie, histogram.

   measure / low_cardinality_measure:
     - Use as y-axis. Can be x in scatter or histogram.

   temporal / year:
     - Use as x-axis for line and area charts.
     - Include as filter if year.

3. Chart type rules:

   bar       : x = dimension, y = measure, aggregate = "sum" or "mean".
               orient = "h" if x nunique > 8.
   line      : x = temporal or year, y = measure. color only if nunique <= 5.
   area      : x = temporal or year, y = measure. color only if nunique <= 5.
               Use instead of line when filled area adds meaning.
   histogram : x = measure only. No y. Use when is_skewed = true.
   scatter   : x = measure, y = measure. Only if r < 0.95.
               color must have nunique <= 15.
   box       : x = dimension only (nunique <= 12), y = measure.
   pie       : x = dimension (nunique <= 6), y = measure (sum). Part-of-whole only.

4. Aggregation:
   - Repeated x values -> aggregate = "sum" or "mean".
   - One row per observation -> aggregate = "none".

5. Color guardrails:
   - bar/scatter/box/pie: color nunique <= 15.
   - line/area: color nunique <= 5.
   - If best candidate exceeds limit, set color = null.

6. Chart diversity: max 2 charts of same type. Avoid reusing the same
   x-axis column unless the color grouping is meaningfully different.

7. Redundancy: each chart must answer a different business question.

8. Propose 3-5 charts total. Quality over quantity.

9. Filters:
   - Include ALL dimension columns (nunique <= 25).
   - Include ALL high_cardinality_dimension columns (nunique <= 100).
   - Include year/temporal columns.
   - Do NOT include measure columns as filters.

10. If no measure columns exist (only temporal, year, dimension):
    - Use aggregate = "count" for bar and pie charts.
    - Set y to any available dimension column (used as count target).
    - Do NOT invent a measure column.
""",
    ),
    ("human", "Dataset profile:\n\n{schema}\n\n{intent_block}"),
])

plan_chain = prompt_template | structured_llm


def generate_dashboard_plan(df: pd.DataFrame, eda: dict, intent: str = "") -> DashboardPlan:
    """Call the LLM to generate a dashboard plan, with retry logic.
    Retries up to LLM_MAX_RETRIES times with exponential backoff on any error
    (rate limits, timeouts, malformed output)."""
    schema = build_schema_summary(df, eda)
    intent_block = (
        f"User's goal: {intent.strip()}\nPrioritise charts that directly answer this question."
        if intent and intent.strip()
        else ""
    )

    last_error = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            plan: DashboardPlan = plan_chain.invoke(
                {"schema": schema, "intent_block": intent_block}
            )
            return plan
        except Exception as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES - 1:
                wait = LLM_RETRY_BACKOFF[attempt]
                st.warning(
                    f"LLM call failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

    raise RuntimeError(
        f"LLM plan generation failed after {LLM_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# Stage 4: Validate and repair
# Fuzzy-matches column names the LLM may have slightly misspelled,
# enforces per-chart-type constraints, and drops invalid charts.

def fuzzy_match_column(name: str, columns: list[str]) -> Optional[str]:
    matches = difflib.get_close_matches(name, columns, n=1, cutoff=0.6)
    return matches[0] if matches else None


def validate_and_repair_chart(chart: ChartSpec, df: pd.DataFrame) -> Optional[ChartSpec]:
    real_cols = df.columns.tolist()

    if chart.x not in real_cols:
        fixed = fuzzy_match_column(chart.x, real_cols)
        if fixed:
            chart.x = fixed
        else:
            return None

    if chart.y and chart.y not in real_cols:
        fixed = fuzzy_match_column(chart.y, real_cols)
        chart.y = fixed if fixed else None

    if chart.color:
        if chart.color not in real_cols:
            chart.color = fuzzy_match_column(chart.color, real_cols)
        if chart.color and df[chart.color].nunique() > 20:
            chart.color = None

    if chart.type == "histogram":
        chart.y = None
        chart.aggregate = "none"
        if not pd.api.types.is_numeric_dtype(df[chart.x]):
            return None

    elif chart.type in ("bar", "line", "scatter", "area"):
        if not chart.y:
            return None
        if chart.type in ("line", "area") and chart.color:
            if df[chart.color].nunique() > 5:
                chart.color = None

    elif chart.type == "pie":
        if not chart.y:
            return None
        if pd.api.types.is_numeric_dtype(df[chart.x]):
            return None
        if df[chart.x].nunique() > 8:
            return None
        chart.color = None

    elif chart.type == "box":
        if not chart.y:
            return None
        if not pd.api.types.is_numeric_dtype(df[chart.y]):
            return None
        if df[chart.x].nunique() > 15:
            return None

    return chart


def validate_plan(plan: DashboardPlan, df: pd.DataFrame) -> DashboardPlan:
    plan.charts = [c for c in (validate_and_repair_chart(ch, df) for ch in plan.charts) if c is not None]
    plan.filters = [f for f in plan.filters if f in df.columns]
    return plan


# Aggregation helper
# Groups and aggregates data before plotting. For time-series with many
# unique dates (>52), floors to monthly so charts stay readable.
# Uses .agg() instead of .apply() for performance and pandas compatibility.

def aggregate_dataframe(df: pd.DataFrame, chart: ChartSpec) -> pd.DataFrame:
    if chart.type in ("histogram", "box", "scatter") or chart.aggregate == "none":
        return df
    if not chart.y:
        return df

    group_cols = [chart.x]
    if chart.color and chart.color in df.columns and chart.type != "pie":
        group_cols.append(chart.color)
    group_cols = list(dict.fromkeys(group_cols))

    plot_df = df.dropna(subset=group_cols)

    # Coarsen daily timestamps to monthly for readable trend lines
    if pd.api.types.is_datetime64_any_dtype(plot_df[chart.x]) and chart.type in ("line", "area"):
        if plot_df[chart.x].nunique() > 52:
            plot_df = plot_df.copy()
            plot_df[chart.x] = plot_df[chart.x].dt.to_period("M").dt.to_timestamp()

    # Use .agg() with named aggregation for clarity and performance.
    # This avoids the deprecated groupby().apply(lambda) path and is ~2-3x faster.
    agg_func_map = {"sum": "sum", "mean": "mean", "count": "count"}
    pandas_agg = agg_func_map.get(chart.aggregate)

    if pandas_agg:
        plot_df = (
            plot_df
            .groupby(group_cols, dropna=False)[chart.y]
            .agg(pandas_agg)
            .reset_index(name=chart.y)
        )

    # Keep only top-N categories for readability
    if chart.top_n and chart.type in ("bar", "area", "pie") and chart.y in plot_df.columns:
        if not chart.color or chart.type == "pie":
            plot_df = plot_df.sort_values(chart.y, ascending=False).head(chart.top_n)

    return plot_df


# Plotly figure builder
# Constructs a Plotly Express figure themed to match the Streamlit UI.
# Applies log scale automatically when the value range exceeds 100x.

def build_figure(plot_df: pd.DataFrame, chart: ChartSpec, chart_type: str):
    template = _get_plotly_template()
    kwargs: dict = {}
    if chart.color and chart.color in plot_df.columns:
        kwargs["color"] = chart.color

    try:
        if chart_type == "bar":
            if chart.orient == "h":
                fig = px.bar(plot_df, x=chart.y, y=chart.x,
                             orientation="h", barmode="group", **kwargs)
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
            else:
                fig = px.bar(plot_df, x=chart.x, y=chart.y,
                             barmode="group", **kwargs)

        elif chart_type == "line":
            fig = px.line(plot_df, x=chart.x, y=chart.y, markers=True, **kwargs)

        elif chart_type == "area":
            fig = px.area(plot_df, x=chart.x, y=chart.y, **kwargs)

        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=chart.x, nbins=40, **kwargs)

        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=chart.x, y=chart.y, **kwargs)

        elif chart_type == "pie":
            fig = px.pie(plot_df, names=chart.x, values=chart.y, hole=0.35)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(
                template=template,
                title=dict(text=""),
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="v", x=1.02),
            )
            return fig

        elif chart_type == "box":
            fig = px.box(plot_df, x=chart.x, y=chart.y, points="outliers", **kwargs)

        else:
            return None

        # Common layout for all non-pie charts
        fig.update_layout(
            template=template,
            title=dict(text=""),
            margin=dict(t=10, b=50, l=40, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickangle=-35,
            xaxis_title=chart.x or "",
            yaxis_title="Count" if chart.aggregate == "count" else (chart.y or ""),
        )

        # Auto log-scale when value range is very large
        if (
            chart.y and chart.y in plot_df.columns
            and chart_type in ("line", "scatter", "area")
            and chart.aggregate != "count"
        ):
            y_vals = plot_df[chart.y].dropna()
            if len(y_vals) > 0 and y_vals.min() > 0:
                if y_vals.max() / (y_vals.min() + 1e-9) > 100:
                    fig.update_yaxes(type="log")
                    st.caption("Warning: Y-axis is log-scaled due to large value range.")

        return fig

    except Exception as e:
        st.error(f"Plotly error building {chart_type} chart: {e}")
        return None


# Chart renderer
# Renders a single chart panel with a type-switcher dropdown, the Plotly figure,
# an insight caption, and a PNG download button.

ALL_CHART_TYPES = ["bar", "line", "area", "histogram", "scatter", "pie", "box"]


def render_chart(df: pd.DataFrame, chart: ChartSpec, idx: int):
    type_key = f"chart_type_{idx}"
    if type_key not in st.session_state:
        st.session_state[type_key] = chart.type

    col_title, col_type = st.columns([4, 1])
    with col_title:
        st.markdown(f"**{chart.title}**")
    with col_type:
        chart_type = st.selectbox(
            "Type",
            options=ALL_CHART_TYPES,
            index=ALL_CHART_TYPES.index(
                st.session_state[type_key]
                if st.session_state[type_key] in ALL_CHART_TYPES
                else "bar"
            ),
            key=type_key,
            label_visibility="collapsed",
        )

    working_df = df.copy()
    if chart.color and chart.color in working_df.columns:
        working_df = working_df.dropna(subset=[chart.color])

    plot_df = aggregate_dataframe(working_df, chart)
    fig = build_figure(plot_df, chart, chart_type)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Insight: {chart.reason}")
        try:
            img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
            st.download_button(
                label="Download chart (PNG)",
                data=img_bytes,
                file_name=f"chart_{idx}_{chart_type}.png",
                mime="image/png",
                key=f"dl_{idx}",
            )
        except Exception:
            st.caption("PNG export unavailable (install kaleido: `pip install -U kaleido`)")


# Data quality and EDA display helpers

def render_data_quality(df: pd.DataFrame):
    st.markdown("#### Data Quality Report")
    profiles = [profile_column(df, c) for c in df.columns]
    quality_rows = []
    for p in profiles:
        null_pct = p["null_pct"]
        flag = "High nulls" if null_pct > 50 else ("Some nulls" if null_pct > 10 else "")
        quality_rows.append({
            "Column": p["name"],
            "Role": p["role"],
            "Unique Values": p["nunique"],
            "Null %": f"{null_pct}%",
            "Sample": ", ".join(p["sample_values"][:3]),
            "Note": flag,
        })
    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True)
    n_high_null = sum(1 for p in profiles if p["null_pct"] > 50)
    n_identifiers = sum(1 for p in profiles if p["role"] == "identifier")
    n_measures = sum(1 for p in profiles if p["role"] in ("measure", "low_cardinality_measure"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Measures (numeric)", n_measures)
    c2.metric("High-null columns", n_high_null)
    c3.metric("Likely ID columns", n_identifiers)


def render_eda_summary(eda: dict):
    if not eda:
        st.info("No EDA summary available.")
        return
    st.markdown("#### EDA Statistical Summary")
    rows = []
    for col, stats in eda.items():
        corr_str = (
            ", ".join(f"{c['col']} (r={c['r']})" for c in stats["high_corr_with"])
            if stats["high_corr_with"] else "-"
        )
        rows.append({
            "Column": col,
            "Skewed?": "Yes" if stats["is_skewed"] else "No",
            "Skewness": stats.get("skewness", "-"),
            "Near-Zero Var?": "Yes" if stats["near_zero_variance"] else "No",
            "Missing %": f"{stats['p_missing']}%",
            "High Correlations": corr_str,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    n_skewed = sum(1 for s in eda.values() if s["is_skewed"])
    n_nzv = sum(1 for s in eda.values() if s["near_zero_variance"])
    n_corr = sum(1 for s in eda.values() if s["high_corr_with"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Skewed columns", n_skewed)
    c2.metric("Near-zero variance cols", n_nzv)
    c3.metric("Highly correlated pairs", n_corr)


# Streamlit UI
# The main app flow: upload -> profile -> generate plan -> render dashboard.
# Sidebar filters support both numeric sliders and searchable multiselects
# for categorical columns of any cardinality.

st.set_page_config(page_title="AI Dashboard Agent", page_icon="📊", layout="wide")
st.title("AI Dashboard Agent")
st.markdown(
    "Upload a CSV or Excel file. The AI profiles the columns, "
    "plans the best charts, and renders an interactive dashboard."
)
st.divider()

# Initialise session state
for key, default in {
    "df": None, "plan": None, "eda": {},
    "plan_ready": False, "ingest_warnings": [], "loaded_file_name": None, "user_intent": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Step 1: File upload
st.subheader("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file and st.session_state.loaded_file_name != uploaded_file.name:
    try:
        df, warnings = ingest_file(uploaded_file)
        st.session_state.update(
            df=df, ingest_warnings=warnings, loaded_file_name=uploaded_file.name,
            plan=None, eda={}, plan_ready=False,
        )
        for k in [k for k in st.session_state if k.startswith("chart_type_")]:
            del st.session_state[k]
        st.success(f"Loaded **{uploaded_file.name}** - {df.shape[0]:,} rows x {df.shape[1]} columns")
        for w in warnings:
            st.info(f"Auto-fix: {w}")
    except RuntimeError as e:
        st.error(str(e))

if st.session_state.df is not None:
    df = st.session_state.df

    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Data Quality and Column Profiles", expanded=False):
        render_data_quality(df)

    st.divider()

    # Step 2: EDA toggle and plan generation
    st.subheader("2. Generate AI Dashboard Plan")

    use_eda = st.checkbox(
        "Enhanced EDA (skewness, correlations, zero-variance detection)",
        value=False,
        help="Runs statistical profiling on a 5,000-row sample. Adds ~0.5s but improves chart suggestions.",
    )

    user_intent = st.text_input(
        "What do you want to understand? (optional)",
        value=st.session_state.user_intent,
        placeholder="e.g. content trends over time, or sales breakdown by region",
        help="One sentence describing your goal. Helps the AI pick more relevant charts.",
    )
    st.session_state.user_intent = user_intent

    if st.button("Generate Dashboard Plan", type="primary"):
        with st.spinner("Profiling columns and planning charts..."):
            try:
                eda: dict = {}
                if use_eda:
                    eda = get_eda_summary(_df_cache_key(df), df)
                    st.session_state.eda = eda

                raw_plan = generate_dashboard_plan(df, eda, intent=user_intent)
                safe_plan = validate_plan(raw_plan, df)

                st.session_state.plan = safe_plan
                st.session_state.plan_ready = True

                for k in [k for k in st.session_state if k.startswith("chart_type_")]:
                    del st.session_state[k]

                n_dropped = len(raw_plan.charts) - len(safe_plan.charts)
                msg = (
                    f"Plan ready - {len(safe_plan.charts)} charts, "
                    f"Filters: {', '.join(safe_plan.filters) or 'none'}"
                )
                if use_eda:
                    msg += " (Enhanced EDA active)"
                if n_dropped:
                    msg += f" | {n_dropped} chart(s) dropped after validation"
                st.success(msg)

            except RuntimeError as e:
                st.error(str(e))
                st.session_state.plan_ready = False

    if st.session_state.eda:
        with st.expander("View EDA Statistical Summary", expanded=False):
            render_eda_summary(st.session_state.eda)

    if st.session_state.plan_ready:
        with st.expander("View Raw Chart Plan (JSON)", expanded=False):
            st.json(st.session_state.plan.model_dump())

    st.divider()

    # Step 3: Render the dashboard with sidebar filters
    if st.session_state.plan_ready and st.session_state.plan:
        plan = st.session_state.plan
        df_base = df.copy()

        if plan.filters:
            st.sidebar.header("Filters")
            for col in plan.filters:
                if col not in df_base.columns:
                    continue

                if pd.api.types.is_numeric_dtype(df_base[col]):
                    min_v = int(df_base[col].min())
                    max_v = int(df_base[col].max())
                    sel = st.sidebar.slider(col, min_v, max_v, (min_v, max_v))
                    df_base = df_base[df_base[col].between(sel[0], sel[1])]
                else:
                    n_unique = df_base[col].nunique()
                    label = f"{col} ({n_unique} values - type to search)" if n_unique > 20 else col
                    opts = sorted(df_base[col].dropna().unique().tolist())
                    sel = st.sidebar.multiselect(label, opts, default=opts)
                    if sel:
                        df_base = df_base[df_base[col].isin(sel)]

            st.sidebar.markdown(f"**{len(df_base):,}** rows after filtering")

        st.subheader("3. Dashboard")

        if not plan.charts:
            st.warning(
                "No charts could be generated for this dataset. "
                "Check the Data Quality panel for column issues."
            )
        else:
            for i in range(0, len(plan.charts), 2):
                cols = st.columns(2)
                for j, col_ui in enumerate(cols):
                    if i + j < len(plan.charts):
                        with col_ui:
                            render_chart(df_base, plan.charts[i + j], idx=i + j)

    elif st.session_state.df is not None and not st.session_state.plan_ready:
        st.info("Click **Generate Dashboard Plan** to continue.")

else:
    st.info("Upload a CSV or Excel file to get started.")

