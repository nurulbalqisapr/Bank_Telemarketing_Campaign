
# ============================================================
# Bank Telemarketing Telemarketing Intelligence App
# Professional Streamlit Portfolio Deployment
# ============================================================
# Required files in the same folder:
# 1. app.py
# 2. bank_cleaned_for_tableau.csv  OR  bank-additional-full.csv
# 3. best_lr_model.pkl
# 4. best_threshold.pkl
# 5. model_columns.pkl
#
# The app still runs without model artifacts.
# In that case, EDA pages are available and the prediction page
# will show a clear instruction to add the model files.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import base64

try:
    import joblib
except Exception:
    joblib = None


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Bank Telemarketing Campaign Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Theme & CSS
# ============================================================

TEAL = "#1F6F8B"
DARK_TEAL = "#184E77"
NAVY = "#102A43"
BLUE = "#2C5282"
LIGHT_BLUE = "#D9E8F5"
ORANGE = "#D69E2E"
GREEN = "#2F855A"
RED = "#B45309"
GRAY = "#5B6B7A"
LIGHT_BG = "#EEF3F8"
BORDER = "#D7E0EA"


st.markdown(
    f"""
    <style>
        /* Main page */
        .stApp {{
            background: linear-gradient(180deg, #EFF4F8 0%, #F7FAFC 46%, #EDF2F7 100%);
            font-family: "Inter", "Segoe UI", sans-serif;
        }}
        .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }}

        /* Hide default noise */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F2942 0%, #102A43 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {{
            color: #EAF3FF;
        }}
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stMultiSelect,
        section[data-testid="stSidebar"] .stNumberInput {{
            color: {NAVY};
        }}

        /* Containers */
        .hero {{
            background: linear-gradient(135deg, #17324D 0%, #102A43 50%, #285E8E 100%);
            padding: 34px 40px;
            border-radius: 26px;
            color: white;
            box-shadow: 0 20px 42px rgba(16, 42, 67, 0.18);
            margin-bottom: 22px;
        }}
        .hero-title {{
            font-size: 42px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 8px;
        }}
        .hero-subtitle {{
            font-size: 17px;
            color: #EAF2F8;
            max-width: 1080px;
            line-height: 1.55;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 34px;
            align-items: center;
        }}
        .hero-visual {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 26px;
            padding: 18px;
            min-height: 270px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .hero-visual img {{
            width: 100%;
            max-width: 330px;
            height: auto;
            display: block;
            filter: drop-shadow(0px 18px 26px rgba(16, 42, 67, 0.22));
        }}
        .welcome-label {{
            color: #F2C94C;
            font-size: 15px;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .hero-note {{
            color: #EAF2F8;
            font-size: 15px;
            line-height: 1.6;
            margin-top: 14px;
            max-width: 980px;
        }}
        @media (max-width: 900px) {{
            .hero-grid {{
                grid-template-columns: 1fr;
            }}
            .hero-visual {{
                min-height: auto;
            }}
        }}

        .pill {{
            display: inline-block;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.18);
            color: white;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 13px;
            margin-right: 8px;
            margin-bottom: 8px;
        }}

        .metric-card {{
            background: rgba(255,255,255,0.97);
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 20px 22px;
            box-shadow: 0 14px 30px rgba(15, 41, 66, 0.08);
            min-height: 128px;
        }}
        .metric-label {{
            color: {GRAY};
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .metric-value {{
            color: {NAVY};
            font-size: 34px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 6px;
        }}
        .metric-help {{
            color: {GRAY};
            font-size: 13px;
            line-height: 1.35;
        }}

        .section-card {{
            background: rgba(255,255,255,0.96);
            border: 1px solid {BORDER};
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: 0 16px 32px rgba(15, 41, 66, 0.07);
            margin-bottom: 18px;
        }}
        .section-title {{
            color: {NAVY};
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 4px;
        }}
        .section-subtitle {{
            color: {GRAY};
            font-size: 14px;
            margin-bottom: 16px;
            line-height: 1.55;
        }}

        .insight-box {{
            background: linear-gradient(180deg, #F5FAFF 0%, #FFFFFF 100%);
            border: 1px solid #D2E4F1;
            border-left: 6px solid {TEAL};
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 16px;
        }}
        .insight-title {{
            color: {DARK_TEAL};
            font-weight: 800;
            font-size: 16px;
            margin-bottom: 6px;
        }}
        .insight-text {{
            color: {NAVY};
            font-size: 14px;
            line-height: 1.55;
        }}

        .warning-box {{
            background: #FFF9ED;
            border: 1px solid #F6DEAE;
            border-left: 6px solid {ORANGE};
            border-radius: 18px;
            padding: 18px 20px;
            color: {NAVY};
            margin-bottom: 16px;
        }}

        .success-box {{
            background: #EFF8F3;
            border: 1px solid #CAE9D6;
            border-left: 6px solid {GREEN};
            border-radius: 18px;
            padding: 18px 20px;
            color: {NAVY};
            margin-bottom: 16px;
        }}

        .danger-box {{
            background: #FFF1F2;
            border: 1px solid #FECDD3;
            border-left: 6px solid {RED};
            border-radius: 18px;
            padding: 18px 20px;
            color: {NAVY};
            margin-bottom: 16px;
        }}

        .small-muted {{
            color: {GRAY};
            font-size: 13px;
            line-height: 1.45;
        }}

        div[data-testid="stMetricValue"] {{
            font-size: 30px;
            color: {NAVY};
        }}

        .stButton button {{
            background: linear-gradient(135deg, #00897B 0%, #00695C 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.75rem 1.25rem;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(0, 137, 123, 0.25);
            width: 100%;
        }}
        
        .stButton button:hover {{
            background: linear-gradient(135deg, #2C7DA0 0%, #1F6F8B 100%);
            color: white;
            border: none;
        }}

        /* Sidebar navigation - highlighted selected page */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 8px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 14px;
            padding: 10px 12px;
            margin: 0 0 6px 0;
            transition: all 0.2s ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
            background: linear-gradient(90deg, rgba(242,201,76,0.18) 0%, rgba(255,255,255,0.08) 100%);
            border: 1px solid rgba(242,201,76,0.30);
            box-shadow: inset 4px 0 0 #F2C94C;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{
            font-weight: 700;
            color: #F8FBFF !important;
            font-size: 1.01rem;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{
            color: #FFFFFF !important;
        }}

        /* Profile cards for case profile section */
        .profile-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-top: 8px;
        }}
        .profile-card {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFD 100%);
            border: 1px solid #D7E3EE;
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            min-height: 110px;
            box-shadow: 0 10px 22px rgba(16, 42, 67, 0.05);
        }}
        .profile-label {{
            color: #5B6B7A;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .profile-value {{
            color: #102A43;
            font-size: 20px;
            font-weight: 800;
            line-height: 1.32;
            word-break: break-word;
        }}
        @media (max-width: 1100px) {{
            .profile-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 700px) {{
            .profile-grid {{
                grid-template-columns: 1fr;
            }}
        }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data Loading & Preparation
# ============================================================

DATA_CANDIDATES = [
    Path("bank_cleaned_for_tableau.csv"),
    Path("bank_cleaned.csv"),
    Path("bank-additional-full.csv"),
    Path("data/bank_cleaned_for_tableau.csv"),
    Path("data/bank-additional-full.csv"),
]

MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_NAME = {m: m.title() for m in MONTH_ORDER}


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load either cleaned CSV or original UCI bank marketing CSV."""
    found_path = None

    for path in DATA_CANDIDATES:
        if path.exists():
            found_path = path
            break

    if found_path is None:
        return pd.DataFrame()

    if found_path.name == "bank-additional-full.csv":
        df = pd.read_csv(found_path, sep=";")
    else:
        df = pd.read_csv(found_path)

    df = df.copy()

    # Basic cleaning based on notebook logic.
    df = df.drop_duplicates()

    if "marital" in df.columns:
        df = df[df["marital"] != "unknown"]

    for col in ["education", "default", "housing", "loan"]:
        if col in df.columns and (df[col] == "unknown").any():
            mode_value = df.loc[df[col] != "unknown", col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].replace("unknown", mode_value.iloc[0])

    return df.reset_index(drop=True)


def prepare_eda_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "y" in df.columns:
        df["y_binary"] = df["y"].map({"no": 0, "yes": 1}).astype(int)

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
            right=True,
        ).astype(str)

    if "campaign" in df.columns:
        df["campaign_group"] = pd.cut(
            df["campaign"],
            bins=[0, 1, 2, 3, 5, 999],
            labels=["1", "2", "3", "4-5", "6+"],
            right=True,
        ).astype(str)

    if "month" in df.columns:
        df["month_label"] = df["month"].map(MONTH_NAME).fillna(df["month"].astype(str).str.title())

    if "euribor3m" in df.columns:
        df["euribor_level"] = pd.cut(
            df["euribor3m"],
            bins=[-999, 1.5, 3.0, 4.5, 999],
            labels=["Low <=1.5", "Mid 1.5-3.0", "Upper 3.0-4.5", "High >4.5"],
        ).astype(str)

    return df


def prepare_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as notebook before prediction."""
    data = raw_df.copy()

    if "duration" in data.columns:
        data = data.drop(columns=["duration"])

    if "pdays" in data.columns:
        data["pdays_contacted"] = (data["pdays"] != 999).astype(int)
        data["pdays"] = data["pdays"].replace(999, 0)

    return data


df_raw = load_dataset()
df = prepare_eda_data(df_raw) if not df_raw.empty else pd.DataFrame()


# ============================================================
# Model Artifact Loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    if joblib is None:
        return None, None, None, ["joblib is not installed"]

    model_path = Path("best_lr_model.pkl")
    threshold_path = Path("best_threshold.pkl")
    columns_path = Path("model_columns.pkl")

    missing = [str(p) for p in [model_path, threshold_path, columns_path] if not p.exists()]
    if missing:
        return None, None, None, missing

    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    columns = joblib.load(columns_path)
    return model, threshold, columns, []


model, best_threshold, model_columns, missing_artifacts = load_model_artifacts()


# ============================================================
# Utility Functions
# ============================================================

def fmt_int(x) -> str:
    try:
        x = int(x)

        if abs(x) >= 1000:
            return f"{x / 1000:.1f}K"

        return str(x)

    except Exception:
        return "-"


def fmt_pct(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.1%}"


def metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def profile_cards(items):
    card_parts = ['<div class="profile-grid">']
    for label, value in items:
        card_parts.append(
            f'<div class="profile-card">'
            f'<div class="profile-label">{label}</div>'
            f'<div class="profile-value">{value}</div>'
            f'</div>'
        )
    card_parts.append('</div>')
    st.markdown("".join(card_parts), unsafe_allow_html=True)



def _image_to_base64(image_path: Path) -> str:
    """Convert local image to base64 for a stable HTML hero image."""
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def render_hero():
    st.markdown(
        """<div class="hero"><div class="hero-title">Bank Telemarketing Campaign Intelligence</div><div class="hero-subtitle">Professional decision-support app for exploring Portugal bank telemarketing campaign results, understanding customer segments, and prioritizing term deposit outreach.</div><div style="margin-top:18px;"><span class="pill">Portugal Bank Telemarketing</span><span class="pill">Interactive EDA</span><span class="pill">Customer Prediction</span><span class="pill">Portfolio-ready Deployment</span></div></div>""",
        unsafe_allow_html=True,
    )


def render_executive_hero():
    img64 = _image_to_base64(Path("assets/bank_hero.jpg"))
    if img64:
        visual_html = f'<img src="data:image/jpg;base64,{img64}" alt="Bank marketing illustration">'
    else:
        visual_html = '<div style="font-size:86px;">🏦</div><div style="color:white;font-weight:800;margin-left:12px;">Bank Illustration</div>'

    st.markdown(
        f"""<div class="hero"><div class="hero-grid"><div class="hero-visual">{visual_html}</div><div><div class="welcome-label">Welcome to the Telemarketing Intelligence App</div><div class="hero-title">Portugal Bank Telemarketing Campaign Intelligence</div><div class="hero-subtitle">Executive dashboard for understanding term deposit campaign performance, exploring customer behavior, and supporting telemarketing prioritization.</div><div style="margin-top:18px;"><span class="pill">Business Case</span><span class="pill">Portugal Bank Campaign</span><span class="pill">Term Deposit Subscription</span><span class="pill">Interactive EDA</span><span class="pill">Customer Prediction</span></div><div class="hero-note">This application starts from business context and interactive EDA, then continues to customer-level prediction to help the telemarketing team focus on higher-potential term deposit prospects.</div></div></div></div>""",
        unsafe_allow_html=True,
    )


def render_about_hero():
    st.markdown(
        """<div class="hero"><div style="display:flex;align-items:center;gap:18px;"><div style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.28);padding:14px 18px;border-radius:18px;font-size:36px;line-height:1;">📌</div><div><div class="welcome-label">Project Documentation</div><div class="hero-title" style="margin-bottom:4px;">About This Project</div><div class="hero-subtitle">Project background, business problem, goals, analytical workflow, app features, and limitations.</div></div></div></div>""",
        unsafe_allow_html=True,
    )


def conversion_summary(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if data.empty or group_col not in data.columns:
        return pd.DataFrame(columns=[group_col, "customers", "subscribers", "conversion_rate"])

    out = (
        data.groupby(group_col, dropna=False)
        .agg(
            customers=("y_binary", "size"),
            subscribers=("y_binary", "sum"),
            conversion_rate=("y_binary", "mean"),
        )
        .reset_index()
    )
    return out


def apply_sidebar_filters(data: pd.DataFrame, key_prefix: str = "eda"):
    st.sidebar.markdown("### Interactive EDA Filters")
    st.sidebar.caption("Default = all customers. Every KPI and chart updates after selection.")

    target_options = ["All", "yes", "no"]
    target = st.sidebar.selectbox("Target Outcome", target_options, index=0, key=f"{key_prefix}_target")

    job_values = sorted(data["job"].dropna().unique().tolist()) if "job" in data else []
    jobs = st.sidebar.multiselect("Job Segment", job_values, default=[], key=f"{key_prefix}_job")

    month_values = [m for m in MONTH_ORDER if m in data["month"].dropna().unique().tolist()] if "month" in data else []
    months = st.sidebar.multiselect(
        "Campaign Month",
        month_values,
        default=[],
        format_func=lambda x: x.title(),
        key=f"{key_prefix}_month",
    )

    contact_values = sorted(data["contact"].dropna().unique().tolist()) if "contact" in data else []
    contacts = st.sidebar.multiselect("Contact Method", contact_values, default=[], key=f"{key_prefix}_contact")

    poutcome_values = sorted(data["poutcome"].dropna().unique().tolist()) if "poutcome" in data else []
    poutcomes = st.sidebar.multiselect("Previous Outcome", poutcome_values, default=[], key=f"{key_prefix}_poutcome")

    campaign_group_values = ["1", "2", "3", "4-5", "6+"]
    campaign_groups = st.sidebar.multiselect(
        "Campaign Frequency",
        [x for x in campaign_group_values if x in data["campaign_group"].dropna().unique().tolist()],
        default=[],
        key=f"{key_prefix}_campaign_group",
    )

    age_group_values = ["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]
    age_groups = st.sidebar.multiselect(
        "Age Group",
        [x for x in age_group_values if x in data["age_group"].dropna().unique().tolist()],
        default=[],
        key=f"{key_prefix}_age_group",
    )

    filtered = data.copy()

    if target != "All":
        filtered = filtered[filtered["y"] == target]
    if jobs:
        filtered = filtered[filtered["job"].isin(jobs)]
    if months:
        filtered = filtered[filtered["month"].isin(months)]
    if contacts:
        filtered = filtered[filtered["contact"].isin(contacts)]
    if poutcomes:
        filtered = filtered[filtered["poutcome"].isin(poutcomes)]
    if campaign_groups:
        filtered = filtered[filtered["campaign_group"].isin(campaign_groups)]
    if age_groups:
        filtered = filtered[filtered["age_group"].isin(age_groups)]

    selected_filters = {
        "Target": target,
        "Job": jobs,
        "Month": months,
        "Contact": contacts,
        "Previous outcome": poutcomes,
        "Campaign frequency": campaign_groups,
        "Age group": age_groups,
    }

    return filtered, selected_filters


def filter_summary_text(selected_filters: dict) -> str:
    parts = []
    for label, val in selected_filters.items():
        if isinstance(val, list):
            if val:
                parts.append(f"{label}: {', '.join([str(v).title() for v in val])}")
        else:
            if val != "All":
                parts.append(f"{label}: {str(val).title()}")
    return " | ".join(parts) if parts else "Default view: all cleaned customers"


# ============================================================
# Plot Functions
# ============================================================

def plot_target_donut(data: pd.DataFrame):
    counts = data["y"].value_counts().reindex(["no", "yes"], fill_value=0).reset_index()
    counts.columns = ["Target", "Count"]

    fig = px.pie(
        counts,
        names="Target",
        values="Count",
        hole=0.62,
        color="Target",
        color_discrete_map={"no": LIGHT_BLUE, "yes": BLUE},
    )
    rate = data["y_binary"].mean() if len(data) else 0
    fig.update_traces(textposition="outside", textinfo="label+percent")
    fig.update_layout(
        title="Target Distribution",
        annotations=[
            dict(
                text=f"<b>{rate:.1%}</b><br>Conversion",
                x=0.5,
                y=0.5,
                font_size=18,
                showarrow=False,
            )
        ],
        height=380,
        margin=dict(t=60, b=20, l=10, r=10),
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def plot_conversion_bar(data: pd.DataFrame, group_col: str, title: str, orientation: str = "h", order=None, top_n=None):
    summary = conversion_summary(data, group_col)
    if summary.empty:
        return go.Figure()

    if order is not None:
        summary[group_col] = pd.Categorical(summary[group_col], categories=order, ordered=True)
        summary = summary.sort_values(group_col)
    else:
        summary = summary.sort_values("conversion_rate", ascending=False)

    if top_n:
        summary = summary.head(top_n)

    if orientation == "h":
        fig = px.bar(
            summary.sort_values("conversion_rate"),
            x="conversion_rate",
            y=group_col,
            orientation="h",
            text=summary.sort_values("conversion_rate")["conversion_rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"customers": ":,", "subscribers": ":,", "conversion_rate": ":.2%"},
            color_discrete_sequence=[BLUE],
        )
        fig.update_xaxes(tickformat=".0%", title="Conversion Rate")
        fig.update_yaxes(title="")
    else:
        fig = px.bar(
            summary,
            x=group_col,
            y="conversion_rate",
            text=summary["conversion_rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"customers": ":,", "subscribers": ":,", "conversion_rate": ":.2%"},
            color_discrete_sequence=[BLUE],
        )
        fig.update_yaxes(tickformat=".0%", title="Conversion Rate")
        fig.update_xaxes(title="")

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(t=60, b=35, l=20, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(color=NAVY),
    )
    return fig


def plot_month_combo(data: pd.DataFrame):
    summary = conversion_summary(data, "month")
    if summary.empty:
        return go.Figure()

    summary["month"] = pd.Categorical(summary["month"], categories=MONTH_ORDER, ordered=True)
    summary = summary.sort_values("month")
    summary["month_label"] = summary["month"].astype(str).str.title()

    fig = go.Figure()
    fig.add_bar(
        x=summary["month_label"],
        y=summary["customers"],
        name="Customer Volume",
        marker_color=LIGHT_BLUE,
        yaxis="y1",
        hovertemplate="Month=%{x}<br>Customers=%{y:,}<extra></extra>",
    )
    fig.add_scatter(
        x=summary["month_label"],
        y=summary["conversion_rate"],
        name="Conversion Rate",
        mode="lines+markers+text",
        marker=dict(color=TEAL, size=8),
        line=dict(color=TEAL, width=3),
        text=[f"{v:.1%}" for v in summary["conversion_rate"]],
        textposition="top center",
        yaxis="y2",
        hovertemplate="Month=%{x}<br>Conversion=%{y:.2%}<extra></extra>",
    )
    fig.update_layout(
        title="Campaign Timing: Conversion by Month",
        height=420,
        margin=dict(t=60, b=35, l=20, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(title="Customer Volume", showgrid=True, gridcolor="#EEF2F6"),
        yaxis2=dict(title="Conversion Rate", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.15),
        font=dict(color=NAVY),
    )
    return fig


def plot_euribor(data: pd.DataFrame):
    order = ["Mid 1.5-3.0", "Low <=1.5", "Upper 3.0-4.5", "High >4.5"]
    summary = conversion_summary(data, "euribor_level")
    if summary.empty:
        return go.Figure()
    summary["euribor_level"] = pd.Categorical(summary["euribor_level"], categories=order, ordered=True)
    summary = summary.sort_values("euribor_level")

    fig = px.bar(
        summary,
        x="euribor_level",
        y="conversion_rate",
        text=summary["conversion_rate"].map(lambda x: f"{x:.1%}"),
        color="euribor_level",
        color_discrete_sequence=["#E15759", ORANGE, "#76B7B2", BLUE],
        hover_data={"customers": ":,", "subscribers": ":,", "conversion_rate": ":.2%"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="Economic Signal: Euribor Level",
        height=380,
        margin=dict(t=60, b=35, l=20, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis_title="",
        yaxis_title="Conversion Rate",
        yaxis_tickformat=".0%",
        font=dict(color=NAVY),
    )
    return fig


# ============================================================
# Pages
# ============================================================


def page_overview():
    if df.empty:
        st.error("Dataset tidak ditemukan. Simpan `bank_cleaned_for_tableau.csv` atau `bank-additional-full.csv` di folder yang sama dengan app.py.")
        return

    total_customers = len(df)
    yes_count = int(df["y_binary"].sum())
    no_count = total_customers - yes_count
    conversion = df["y_binary"].mean()

    render_executive_hero()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header(
        "Business Case",
        "This opening page explains the business context, the dataset, the problem statement, and the project goals.",
    )

    st.markdown(
        """
        This project uses a **Portugal bank marketing campaign dataset**. The dataset describes the results of
        bank marketing campaigns conducted mostly through **direct phone calls**. The campaign offered bank
        clients a **term deposit product**.

        If the client agreed to place a term deposit after the marketing effort, the target variable `y` is marked as
        **yes**. Otherwise, it is marked as **no**.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Cleaned Customers", fmt_int(total_customers), "Rows used after cleaning")
    with c2:
        metric_card("Subscribed Customers", fmt_int(yes_count), "Customers with y = yes")
    with c3:
        metric_card("Baseline Conversion", fmt_pct(conversion), "Overall subscription rate")
    with c4:
        metric_card("Not Subscribed", fmt_int(no_count), "Customers with y = no")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Business Problem", "Why the bank needs a more data-driven telemarketing approach.")
        st.markdown(
            """
            <div class="warning-box">
                Telemarketing campaigns require cost, time, and agent effort. If all customers are contacted
                without any selection or prioritization process, the bank may spend resources on customers who
                have a low probability of subscribing to a term deposit.
                <br><br>
                The dataset also shows that subscribed customers are much fewer than non-subscribed customers.
                This means the campaign conversion rate is relatively low, so the bank needs a better way to
                understand which customers are more likely to subscribe and which customers should be prioritized.
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_header("Problem Statement", "The main question answered by this project.")
        st.markdown(
            """
            <div class="insight-box">
                <div class="insight-title">Main Project Question</div>
                <div class="insight-text">
                    How can the bank identify the key factors that influence customers' tendency to subscribe
                    to a term deposit and predict which customers should be prioritized in telemarketing
                    campaigns to improve telemarketing efficiency?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Project Goals", "Goals are aligned with the notebook business problem.")

        st.markdown(
            """
            <div class="success-box">
                <b>1. Identify key factors</b><br>
                Analyze customer profile, campaign information, previous campaign history, and economic
                indicators that are associated with a higher tendency to subscribe.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="success-box">
                <b>2. Build a classification model</b><br>
                Predict whether a customer is likely to subscribe to a term deposit or not.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="success-box">
                <b>3. Support telemarketing prioritization</b><br>
                Help the bank prioritize high-potential customers, reduce inefficient calls to low-potential
                customers, and improve telemarketing efficiency.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Case Profile", "Quick business context for portfolio readers.")

    profile_cards(
        [
            ("Business Topic", "Term Deposit Telemarketing Campaign"),
            ("Telemarketing Channel", "Direct Phone Calls"),
            ("Target Variable", "y: Yes / No"),
            ("Modeling Type", "Binary Classification"),
        ]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header(
        "How This App Helps",
        "The app follows the same logic as the project: understand the data first, then use prediction to support telemarketing prioritization.",
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#0B1F3A; margin-top:0;">Understand Conversion Patterns</h4>
                <p style="color:#667085; line-height:1.5;">
                    Use EDA to identify factors related to subscription behavior, such as job, contact method,
                    campaign timing, previous outcome, and economic indicators.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#0B1F3A; margin-top:0;">Predict Subscription Potential</h4>
                <p style="color:#667085; line-height:1.5;">
                    Use the classification model to estimate whether a customer is likely to subscribe to a
                    term deposit or not.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#0B1F3A; margin-top:0;">Prioritize Campaign Outreach</h4>
                <p style="color:#667085; line-height:1.5;">
                    Help marketing teams focus on high-potential customers and reduce inefficient calls to
                    low-potential customers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Strategic Value</div>
            <div class="section-subtitle">
                This application helps translate the notebook analysis into a practical decision-support tool.
                Stakeholders can explore the campaign data, understand what factors are associated with subscription,
                and use customer-level prediction to support telemarketing prioritization.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_eda():
    render_hero()

    if df.empty:
        st.error("Dataset tidak ditemukan. Simpan CSV di folder app.")
        return

    filtered, selected_filters = apply_sidebar_filters(df, "eda_page")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header(
        "Interactive EDA Dashboard",
        "Use the filters in the sidebar. Default view shows all customers; selected view updates all KPI cards and charts.",
    )
    st.info(filter_summary_text(selected_filters))

    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan kombinasi filter ini. Coba longgarkan filter.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    selected_customers = len(filtered)
    selected_yes = int(filtered["y_binary"].sum())
    selected_rate = filtered["y_binary"].mean()
    selected_no = selected_customers - selected_yes
    call_efficiency_note = "High conversion segment" if selected_rate >= df["y_binary"].mean() else "Below baseline segment"

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Selected Customers", fmt_int(selected_customers), "Segment size after filters")
    with k2:
        metric_card("Selected Yes Count", fmt_int(selected_yes), "Subscribers in selected view")
    with k3:
        metric_card("Conversion Rate", fmt_pct(selected_rate), "Selected segment conversion")
    with k4:
        metric_card("Selected No Count", fmt_int(selected_no), "Non-subscribers in segment")
    with k5:
        metric_card("Decision Signal", call_efficiency_note, "Compared with baseline conversion")

    st.markdown("</div>", unsafe_allow_html=True)

    top_left, top_mid, top_right = st.columns([0.95, 1.05, 1.05])
    with top_left:
        st.plotly_chart(plot_target_donut(filtered), use_container_width=True)
    with top_mid:
        st.plotly_chart(
            plot_conversion_bar(
                filtered,
                "poutcome",
                "Previous Campaign Outcome",
                orientation="h",
                order=["nonexistent", "failure", "success"],
            ),
            use_container_width=True,
        )
    with top_right:
        st.plotly_chart(
            plot_conversion_bar(filtered, "contact", "Contact Method", orientation="v", order=["telephone", "cellular"]),
            use_container_width=True,
        )

    bottom_left, bottom_mid, bottom_right = st.columns([1.25, 1, 1])
    with bottom_left:
        st.plotly_chart(plot_month_combo(filtered), use_container_width=True)
    with bottom_mid:
        st.plotly_chart(plot_conversion_bar(filtered, "job", "Job Segment Conversion", orientation="h", top_n=8), use_container_width=True)
    with bottom_right:
        st.plotly_chart(plot_euribor(filtered), use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Selected Segment Table", "Use this table to validate the filtered customer group.")
    preview_cols = [
        "age", "job", "marital", "education", "contact", "month", "campaign",
        "pdays", "previous", "poutcome", "euribor3m", "y",
    ]
    preview_cols = [c for c in preview_cols if c in filtered.columns]
    st.dataframe(filtered[preview_cols].head(200), use_container_width=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)


def page_prediction():
    render_hero()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header(
        "Customer Subscription Prediction",
        "Use this page after exploring EDA. The prediction tool uses the exported model artifacts from the notebook.",
    )

    if missing_artifacts:
        st.markdown(
            f"""
            <div class="warning-box">
                <b>Model artifacts belum ditemukan.</b><br>
                Simpan file berikut di folder yang sama dengan app.py:<br>
                <code>best_lr_model.pkl</code>, <code>best_threshold.pkl</code>, dan <code>model_columns.pkl</code>.<br><br>
                Missing: {', '.join(missing_artifacts)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            """
# Jalankan bagian export di notebook:
import joblib

joblib.dump(best_lr_model, "best_lr_model.pkl")
joblib.dump(float(best_threshold), "best_threshold.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
            """,
            language="python",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        """
        <div class="success-box">
            <b>Model ready.</b> Input below represents customer profile, campaign information,
            previous campaign history, and economic indicators available before making a campaign decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("Dataset tidak ditemukan, sehingga default input menggunakan nilai umum.")
        source = pd.DataFrame()
    else:
        source = df.copy()

    def values(col, fallback):
        if not source.empty and col in source.columns:
            return sorted(source[col].dropna().astype(str).unique().tolist())
        return fallback

    def median_value(col, fallback):
        if not source.empty and col in source.columns:
            return float(source[col].median())
        return fallback

    with st.form("prediction_form"):
        st.markdown("#### Customer Profile")
        a, b, c = st.columns(3)

        with a:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox(
                "Job",
                values("job", ["admin.", "blue-collar", "technician", "services", "management", "retired", "entrepreneur", "self-employed", "housemaid", "unemployed", "student", "unknown"]),
                index=0,
            )
            marital = st.selectbox("Marital Status", values("marital", ["married", "single", "divorced"]), index=0)
            education = st.selectbox(
                "Education",
                values("education", ["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y", "basic.6y", "illiterate"]),
                index=0,
            )

        with b:
            default = st.selectbox("Credit Default", values("default", ["no", "yes"]), index=0)
            housing = st.selectbox("Housing Loan", values("housing", ["yes", "no"]), index=0)
            loan = st.selectbox("Personal Loan", values("loan", ["no", "yes"]), index=0)
            contact = st.selectbox("Contact Method", values("contact", ["cellular", "telephone"]), index=0)

        with c:
            month = st.selectbox("Campaign Month", [m for m in MONTH_ORDER if m in values("month", MONTH_ORDER)] or MONTH_ORDER, index=0)
            day_of_week = st.selectbox("Day of Week", values("day_of_week", ["mon", "tue", "wed", "thu", "fri"]), index=0)
            campaign = st.number_input("Contacts During Current Campaign", min_value=1, max_value=60, value=1)
            previous = st.number_input("Previous Contacts", min_value=0, max_value=10, value=0)

        st.markdown("#### Previous Campaign & Economic Context")
        d, e, f = st.columns(3)

        with d:
            pdays_original = st.number_input(
                "Days Since Previous Contact",
                min_value=0,
                max_value=999,
                value=999,
                help="Use 999 if the customer was not previously contacted.",
            )
            poutcome = st.selectbox("Previous Campaign Outcome", values("poutcome", ["nonexistent", "failure", "success"]), index=0)

        with e:
            emp_var_rate = st.number_input("Employment Variation Rate", value=median_value("emp.var.rate", 1.1), format="%.3f")
            cons_price_idx = st.number_input("Consumer Price Index", value=median_value("cons.price.idx", 93.994), format="%.3f")

        with f:
            cons_conf_idx = st.number_input("Consumer Confidence Index", value=median_value("cons.conf.idx", -36.4), format="%.3f")
            euribor3m = st.number_input("Euribor 3 Month Rate", value=median_value("euribor3m", 4.857), format="%.3f")
            nr_employed = st.number_input("Number of Employees", value=median_value("nr.employed", 5191.0), format="%.1f")

        submit = st.form_submit_button("Analyze Customer Potential")

    st.markdown("</div>", unsafe_allow_html=True)

    input_data = pd.DataFrame(
        {
            "age": [age],
            "job": [job],
            "marital": [marital],
            "education": [education],
            "default": [default],
            "housing": [housing],
            "loan": [loan],
            "contact": [contact],
            "month": [month],
            "day_of_week": [day_of_week],
            "campaign": [campaign],
            "pdays": [pdays_original],
            "previous": [previous],
            "poutcome": [poutcome],
            "emp.var.rate": [emp_var_rate],
            "cons.price.idx": [cons_price_idx],
            "cons.conf.idx": [cons_conf_idx],
            "euribor3m": [euribor3m],
            "nr.employed": [nr_employed],
        }
    )

    input_features = prepare_model_features(input_data)

    # Match model training column order.
    try:
        input_features = input_features[model_columns]
    except Exception as e:
        st.error(f"Input columns do not match model_columns.pkl: {e}")
        st.stop()

    with st.expander("Customer Data Sent to Model", expanded=False):
        st.dataframe(input_features, use_container_width=True)

    if submit:
        probability = float(model.predict_proba(input_features)[:, 1][0])
        decision = 1 if probability >= best_threshold else 0

        if probability >= best_threshold:
            potential_label = "High Potential"
            box_class = "success-box"
            recommendation = "Prioritize this customer for telemarketing outreach."
        elif probability >= 0.15:
            potential_label = "Middle Potential"
            box_class = "warning-box"
            recommendation = "Contact this customer if campaign resources are still available."
        else:
            potential_label = "Low Potential"
            box_class = "danger-box"
            recommendation = "Do not prioritize this customer first. Focus on higher-potential customers."

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Prediction Result", "The result is converted into a practical marketing decision.")

        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            metric_card("Subscription Probability", fmt_pct(probability), "Predicted probability of y = yes")
        with r2:
            metric_card("Optimized Threshold", f"{best_threshold:.2f}", "Threshold selected from F3 optimization")
        with r3:
            metric_card("Decision", potential_label, recommendation)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Subscription Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": TEAL},
                    "steps": [
                        {"range": [0, 15], "color": "#FEE2E2"},
                        {"range": [15, best_threshold * 100], "color": "#FEF3C7"},
                        {"range": [best_threshold * 100, 100], "color": "#DCFCE7"},
                    ],
                    "threshold": {
                        "line": {"color": RED, "width": 4},
                        "thickness": 0.8,
                        "value": best_threshold * 100,
                    },
                },
            )
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
            <div class="{box_class}">
                <b>Recommended Action:</b> {recommendation}<br>
                <span class="small-muted">
                    This prediction should be used as a prioritization tool, not as the only decision factor.
                    Marketing teams should still consider campaign capacity, customer relationship policy, and business timing.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)





def page_about():
    render_about_hero()

    st.markdown("## Project Background")
    st.markdown(
        """
        This project analyzes the result of **Portugal bank marketing campaigns** conducted mostly through
        direct phone calls. The campaign offered customers a **term deposit product**, and the target variable
        shows whether the customer subscribed or not.

        The project is built to support the bank in understanding customer subscription behavior and improving
        telemarketing campaign efficiency through exploratory analysis and binary classification modeling.
        """
    )

    st.markdown("## Business Problem")
    st.info(
        """
        Telemarketing campaigns require cost, time, and agent effort. If all customers are contacted without
        selection or prioritization, the bank may spend resources on customers who have a low probability of
        subscribing. Since the number of subscribed customers is much lower than non-subscribed customers,
        the bank needs a better strategy to understand who is more likely to subscribe.
        """
    )

    st.markdown("## Problem Statement")
    st.markdown(
        """
        <div class="insight-box">
            <div class="insight-title">Main Project Question</div>
            <div class="insight-text">
                How can the bank identify the key factors that influence customers' tendency to subscribe to a
                term deposit and predict which customers should be prioritized in telemarketing campaigns to
                improve telemarketing efficiency?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Project Goals")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#0B1F3A; margin-top:0;">Identify Key Factors</h4>
                <p style="color:#667085; line-height:1.5;">
                    Analyze customer profile, campaign information, previous campaign history, and economic
                    indicators that are associated with term deposit subscription.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#0B1F3A; margin-top:0;">Build Classification Model</h4>
                <p style="color:#667085; line-height:1.5;">
                    Build a binary classification model that predicts whether a customer is likely to subscribe
                    to a term deposit or not.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#0B1F3A; margin-top:0;">Improve Campaign Efficiency</h4>
                <p style="color:#667085; line-height:1.5;">
                    Help the bank prioritize high-potential customers and reduce inefficient calls to
                    low-potential customers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Dataset used:** Portugal Bank Marketing Campaign dataset  
            **Unit of analysis:** One row represents one client contact record  
            **Target variable:** `y`, whether the client subscribed to a term deposit  
            **Target classes:** `yes` and `no`  
            **Main task:** Binary classification and campaign analysis
            """
        )

    with col2:
        st.markdown(
            """
            **Key feature groups:**
            - Customer profile: age, job, marital status, education
            - Financial condition: default, housing loan, personal loan
            - Current campaign: contact method, month, day, campaign frequency
            - Previous campaign: previous contact, pdays, and poutcome
            - Economic indicators: euribor3m, employment variation rate, consumer confidence index
            """
        )

    st.markdown("## Analytical Approach")
    st.markdown(
        """
        The project uses **supervised learning with binary classification** because the target variable has
        two classes: customers who subscribe to a term deposit and customers who do not subscribe.

        The workflow starts with **Exploratory Data Analysis (EDA)** to understand the relationship between
        customer characteristics, campaign information, previous campaign history, economic indicators, and
        subscription behavior. After that, classification models are compared using the same preprocessing
        pipeline, and the best model is selected using a metric that is suitable for imbalanced data.
        """
    )

    st.markdown("## Evaluation Metric")
    st.warning(
        """
        Because the dataset is imbalanced, accuracy is not used as the main metric. This project focuses on
        **F-beta Score (β > 1) for the positive class (`yes`)**, supported by ROC-AUC, PR-AUC, and confusion matrix.
        This metric is aligned with the goal of identifying customers who are more likely to subscribe.
        """
    )

    st.markdown("## Key App Features")
    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            **Executive Overview**
            - Explains the business case and dataset context
            - Shows the business problem, problem statement, and project goals
            - Summarizes how the app supports telemarketing prioritization

            **Interactive EDA**
            - Filter by target outcome, job, month, contact method, previous outcome, campaign frequency, and age group
            - KPI cards and charts update based on selected filters
            - Helps stakeholders understand factors associated with subscription behavior
            """
        )

    with feature_col2:
        st.markdown(
            """
            **Customer Prediction**
            - Input individual customer and campaign information
            - Estimate whether the customer is likely to subscribe
            - Support prioritization of high-potential customers

            **Portfolio Presentation**
            - Clean business-first layout
            - Clear separation between EDA and prediction
            - Professional Streamlit deployment interface
            """
        )

    st.markdown("## Project Limitation")
    st.warning(
        """
        The prediction output is intended as a decision-support tool, not an absolute decision rule.
        Actual campaign results may vary depending on customer behavior, campaign execution, product offering,
        timing, and business policy. The `duration` feature also needs careful treatment because it may cause
        data leakage if used before a call is completed.
        """
    )

    st.markdown("## Tools Used")
    st.markdown(
        """
        - **Python** for data analysis and modeling  
        - **Pandas & NumPy** for data manipulation  
        - **Plotly** for interactive visualization  
        - **Scikit-learn** for modeling workflow  
        - **Streamlit** for deployment and web application interface  
        """
    )


# ============================================================
# Sidebar Navigation
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding: 6px 0 22px 0;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="
                width:36px;
                height:36px;
                border-radius:10px;
                background: linear-gradient(135deg, #2C7DA0 0%, #1F6F8B 100%);
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-size:18px;
                flex-shrink:0;
                box-shadow: 0 10px 18px rgba(0,0,0,0.18);
            ">🏦</div>
            <div>
                <div style="font-size: 19px; font-weight: 900; color: white; line-height:1.15; white-space:nowrap;">Bank Telemarketing</div>
                <div style="font-size: 12px; color: #B9D7EA;">Campaign Intelligence App</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Interactive EDA",
        "Customer Prediction",
        "About Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Portfolio-ready telemarketing decision-support app with Executive Overview, Interactive EDA, and Customer Prediction.")


# ============================================================
# Router
# ============================================================

if page == "Executive Overview":
    page_overview()
elif page == "Interactive EDA":
    page_eda()
elif page == "Customer Prediction":
    page_prediction()
else:
    page_about()
