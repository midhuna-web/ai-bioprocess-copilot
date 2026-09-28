import streamlit as st
import pandas as pd
import json
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ScaleWise | AI Bioprocess Copilot",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return {}

    with open(path, "r") as f:
        return json.load(f)


def load_csv(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


dashboard_data = load_json(
    "ai_copilot_dashboard_data.json"
)

operating_window = load_csv(
    "model2_operating_window.csv"
)

representative_condition = load_csv(
    "model2_representative_condition.csv"
)

guardian_df = load_csv(
    "model3_guardian_data.csv"
)

threshold_data = load_json(
    "model3_alert_threshold.json"
)

# ============================================================
# EXTRACT DASHBOARD DATA
# ============================================================

readiness = dashboard_data.get(
    "readiness",
    {}
)

target_scale_data = dashboard_data.get(
    "target_scale",
    {}
)

biology = dashboard_data.get(
    "biology",
    {}
)

engineering = dashboard_data.get(
    "engineering",
    {}
)

risk_guardian = dashboard_data.get(
    "risk_guardian",
    {}
)

ai_recommendation = dashboard_data.get(
    "ai_recommendation",
    {}
)

overall_score = readiness.get(
    "overall_score",
    0
)

status = readiness.get(
    "status",
    "VALIDATE"
)

categories = readiness.get(
    "categories",
    {}
)

target_scale = target_scale_data.get(
    "target_scale_L",
    20
)

representative = target_scale_data.get(
    "representative_condition",
    {}
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 16px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 15px;
}

.copilot-box {
    padding: 22px;
    border-radius: 14px;
    background-color: #f5f7fa;
    border-left: 5px solid #4c78a8;
    margin-bottom: 15px;
}

.status-box {
    padding: 14px;
    border-radius: 12px;
    background-color: #f5f7fa;
    text-align: center;
}

.small-note {
    font-size: 13px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🧬 ScaleWise — AI Copilot for Scalable Cell-Culture Bioprocess Design'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the biology • Optimize the operating window • Monitor scale-up risk'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Process Configuration")

selected_scale = st.sidebar.number_input(
    "Target Scale (L)",
    min_value=1.0,
    max_value=10000.0,
    value=float(target_scale),
    step=1.0
)

process_stage = st.sidebar.selectbox(
    "Process Stage",
    [
        "Scale-Up Assessment",
        "Process Validation",
        "Production Monitoring"
    ]
)

st.sidebar.info(
    f"Dashboard target scale: {target_scale:.0f} L"
)

st.sidebar.warning(
    "Prototype dashboard using synthetic/demo data. "
    "Process limits and recommendations require experimental validation."
)

# ============================================================
# PROCESS READINESS
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Process Readiness</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Target Scale",
        f"{selected_scale:.0f} L"
    )

with col2:
    st.metric(
        "Overall Readiness",
        f"{overall_score:.2f} / 100"
    )

with col3:
    st.metric(
        "Status",
        status
    )

# ============================================================
# READINESS CATEGORIES
# ============================================================

st.markdown(
    '<div class="section-title">📊 Readiness Categories</div>',
    unsafe_allow_html=True
)

category_names = [
    ("Biological Condition", "Biological Condition"),
    ("Oxygen Transfer", "Oxygen Transfer"),
    ("Hydrodynamic Condition", "Hydrodynamic Condition"),
    ("Process Control", "Process Control"),
    ("Model Reliability", "Model Reliability")
]

cols = st.columns(5)

for col, (display_name, key) in zip(
    cols,
    category_names
):

    value = categories.get(key, "--")

    with col:

        if isinstance(value, (int, float)):
            st.metric(
                display_name,
                f"{value:.1f} / 100"
            )
        else:
            st.metric(
                display_name,
                str(value)
            )

# ============================================================
# OPERATING WINDOW
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Recommended Operating Window</div>',
    unsafe_allow_html=True
)

if not operating_window.empty:

    st.dataframe(
        operating_window,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Operating-window data not available."
    )

# ============================================================
# REPRESENTATIVE CONDITION
# ============================================================

st.markdown(
    '<div class="section-title">📍 Representative Target-Scale Condition</div>',
    unsafe_allow_html=True
)

if representative:

    rep_items = []

    for key, value in representative.items():

        if pd.isna(value):
            continue

        if isinstance(value, (int, float)):

            rep_items.append(
                (
                    key,
                    f"{value:.3f}"
                )
            )

    rep_cols = st.columns(
        min(4, len(rep_items))
    )

    for col, (key, value) in zip(
        rep_cols,
        rep_items
    ):

        with col:
            st.metric(
                key.replace("_", " "),
                value
            )

# ============================================================
# BIOLOGICAL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🧫 Predicted Biological Performance</div>',
    unsafe_allow_html=True
)

b1, b2, b3, b4 = st.columns(4)

with b1:
    st.metric(
        "VCD",
        f"{biology.get('VCD_million_cells_mL', 0):.3f} M cells/mL"
    )

with b2:
    st.metric(
        "Viability",
        f"{biology.get('viability_percent', 0):.2f}%"
    )

with b3:
    st.metric(
        "Growth Rate",
        f"{biology.get('growth_rate_per_h', 0):.4f} /h"
    )

with b4:
    st.metric(
        "Lactate",
        f"{biology.get('lactate_g_L', 0):.3f} g/L"
    )

# ============================================================
# ENGINEERING INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">🔬 Scale-Up Engineering Indicators</div>',
    unsafe_allow_html=True
)

e1, e2, e3 = st.columns(3)

with e1:
    st.metric(
        "P/V",
        f"{engineering.get('PV_W_L', 0):.3f} W/L"
    )

with e2:
    st.metric(
        "kLa",
        f"{engineering.get('kLa_per_h', 0):.3f} /h"
    )

with e3:
    st.metric(
        "Mixing Time",
        f"{engineering.get('mixing_time_s', 0):.1f} s"
    )

e4, e5, e6 = st.columns(3)

with e4:
    st.metric(
        "Tip Speed",
        f"{engineering.get('tip_speed_m_s', 0):.3f} m/s"
    )

with e5:

    re_value = representative.get(
        "reynolds_number",
        0
    )

    st.metric(
        "Reynolds Number",
        f"{float(re_value):,.0f}"
    )

with e6:
    st.metric(
        "DO",
        f"{engineering.get('DO_percent', 0):.2f}%"
    )

# ============================================================
# REAL-TIME RISK GUARDIAN
# ============================================================

st.markdown(
    '<div class="section-title">🛡️ Real-Time Scale-Up Risk Guardian</div>',
    unsafe_allow_html=True
)

if not guardian_df.empty:

    latest = guardian_df.iloc[-1]

    probability = float(
        latest.get(
            "failure_probability",
            0
        )
    )

    risk_level = str(
        latest.get(
            "risk_level",
            "UNKNOWN"
        )
    )

    confidence = float(
        latest.get(
            "model_confidence",
            0
        )
    )

    warning_count = int(
        latest.get(
            "warning_signal_count",
            0
        )
    )

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Deviation Probability",
            f"{probability * 100:.1f}%"
        )

    with r2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with r3:
        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )

    with r4:
        st.metric(
            "Warning Signals",
            warning_count
        )

    # --------------------------------------------------------
    # RISK REASON
    # --------------------------------------------------------

    risk_reason = str(
        latest.get(
            "risk_reason",
            "No active risk driver identified."
        )
    )

    recommended_action = str(
        latest.get(
            "recommended_action",
            "Continue monitoring."
        )
    )

    st.warning(
        f"⚠️ **Risk signal:** {risk_reason}"
    )

    st.info(
        f"🔧 **Recommended action:** {recommended_action}"
    )

    st.caption(
        "Model confidence represents classification confidence "
        "and should not be interpreted as biological certainty."
    )

else:

    st.info(
        "Risk Guardian data not available."
    )

# ============================================================
# RISK TREND
# ============================================================

if not guardian_df.empty:

    st.markdown(
        '<div class="section-title">📈 Risk Guardian Trend</div>',
        unsafe_allow_html=True
    )

    trend_df = guardian_df.copy()

    if "timestamp" in trend_df.columns:

        trend_df["timestamp"] = pd.to_datetime(
            trend_df["timestamp"],
            errors="coerce"
        )

        trend_df = trend_df.dropna(
            subset=["timestamp"]
        )

        if not trend_df.empty:

            trend_df = trend_df.set_index(
                "timestamp"
            )

            chart_columns = [
                c for c in [
                    "failure_probability",
                    "DO_percent",
                    "oxygen_margin"
                ]
                if c in trend_df.columns
            ]

            if chart_columns:

                st.line_chart(
                    trend_df[chart_columns],
                    use_container_width=True
                )

# ============================================================
# AI COPILOT ASSESSMENT
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Copilot Assessment</div>',
    unsafe_allow_html=True
)

recommendation = ai_recommendation.get(
    "recommendation",
    "Use the recommended operating region as a starting point for validation."
)

st.markdown(
    f"""
    <div class="copilot-box">

    <b>AI Copilot Recommendation</b>

    <br><br>

    {recommendation}

    <br><br>

    <b>Target scale:</b> {target_scale:.0f} L

    <br><br>

    <b>Process status:</b> {status}

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# RISK EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Why is the Process at Risk?</div>',
    unsafe_allow_html=True
)

if not guardian_df.empty:

    latest = guardian_df.iloc[-1]

    warning_columns = [
        ("DO", "DO_warning"),
        ("Oxygen Transfer", "oxygen_warning"),
        ("Viability", "viability_warning"),
        ("pH", "pH_warning"),
        ("Mixing", "mixing_warning")
    ]

    active_warnings = []

    for label, column in warning_columns:

        if column in guardian_df.columns:

            if bool(latest[column]):
                active_warnings.append(label)

    if active_warnings:

        st.write(
            "**Active risk indicators:** "
            + ", ".join(active_warnings)
        )

    else:

        st.success(
            "No active rule-based warning signals detected "
            "in the displayed Guardian observation."
        )

# ============================================================
# DATA DISCLAIMER
# ============================================================

st.markdown("---")

st.caption(
    "ScaleWise | AI Copilot for Scalable Cell-Culture Bioprocess Design"
)

st.caption(
    "Prototype / synthetic-data demonstration. "
    "Operating limits, predictions and risk signals require "
    "experimental validation before production use."
)
