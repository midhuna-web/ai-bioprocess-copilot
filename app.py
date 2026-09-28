import streamlit as st
import pandas as pd
import json
import os
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ScaleWise | AI Bioprocess Copilot",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# DATA LOADERS
# ============================================================

def load_json(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_csv(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


dashboard_data = load_json(
    "ai_copilot_dashboard_data.json"
)

operating_window_raw = load_csv(
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
# HELPER FUNCTIONS
# ============================================================

def first_available(dictionary, keys, default=None):

    if not isinstance(dictionary, dict):
        return default

    for key in keys:
        if key in dictionary:
            return dictionary[key]

    return default


def clean_number(value, decimals=2):

    try:
        value = float(value)

        if np.isnan(value):
            return "--"

        return f"{value:.{decimals}f}"

    except Exception:
        return "--"


def get_rep_value(row, keys, default=None):

    if row is None:
        return default

    for key in keys:

        if key in row.index:

            value = row[key]

            if pd.notna(value):
                return value

    return default


# ============================================================
# DASHBOARD DATA EXTRACTION
# ============================================================

readiness = dashboard_data.get(
    "readiness",
    {}
)

categories = readiness.get(
    "categories",
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

ai_recommendation = dashboard_data.get(
    "ai_recommendation",
    {}
)

overall_score = first_available(
    readiness,
    ["overall_score", "score"],
    71.65
)

status = first_available(
    readiness,
    ["status"],
    "VALIDATE"
)

target_scale = first_available(
    target_scale_data,
    ["target_scale_L", "target_scale"],
    20
)


# ============================================================
# REPRESENTATIVE CONDITION
# ============================================================

if not representative_condition.empty:

    rep_row = representative_condition.iloc[0]

else:

    rep_row = pd.Series(dtype=object)


# ============================================================
# TARGET SCALE
# ============================================================

# Current prototype is based on the recovered 20 L scenario.
SUPPORTED_SCALE = float(target_scale)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 16px;
    color: #8b949e;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.copilot-box {
    padding: 24px;
    border-radius: 14px;
    background-color: #f5f7fa;
    border-left: 6px solid #4c78a8;
    margin-bottom: 15px;
    color: #111827 !important;
}

.copilot-box b {
    color: #111827 !important;
}

.copilot-box p {
    color: #111827 !important;
}

.demo-note {
    padding: 12px 15px;
    border-radius: 10px;
    background-color: #fff7d6;
    color: #4b3b00;
    border: 1px solid #e5d58a;
    margin-bottom: 18px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    🧬 ScaleWise — AI Copilot for Scalable Cell-Culture Bioprocess Design
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predict the biology • Optimize the operating window • Monitor scale-up risk
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Process Configuration")

st.sidebar.metric(
    "Demo Target Scale",
    f"{SUPPORTED_SCALE:.0f} L"
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
    "Current prototype scenario: "
    f"{SUPPORTED_SCALE:.0f} L target scale."
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

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Target Scale",
        f"{SUPPORTED_SCALE:.0f} L"
    )

with c2:

    st.metric(
        "Overall Readiness",
        f"{float(overall_score):.2f} / 100"
    )

with c3:

    st.metric(
        "Status",
        str(status)
    )


# ============================================================
# READINESS CATEGORIES
# ============================================================

st.markdown(
    '<div class="section-title">📊 Readiness Categories</div>',
    unsafe_allow_html=True
)

category_keys = [
    "Biological Condition",
    "Oxygen Transfer",
    "Hydrodynamic Condition",
    "Process Control",
    "Model Reliability"
]

category_cols = st.columns(5)

for col, category_name in zip(
    category_cols,
    category_keys
):

    category_value = categories.get(
        category_name,
        {}
    )

    # Categories are dictionaries containing score/weight/details.
    if isinstance(category_value, dict):

        score = first_available(
            category_value,
            ["score", "value", "category_score"],
            None
        )

    else:

        score = category_value

    with col:

        if score is not None:

            try:

                st.metric(
                    category_name,
                    f"{float(score):.1f} / 100"
                )

            except Exception:

                st.metric(
                    category_name,
                    str(score)
                )

        else:

            st.metric(
                category_name,
                "--"
            )


# ============================================================
# OPERATING WINDOW
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Recommended Operating Window</div>',
    unsafe_allow_html=True
)

if not operating_window_raw.empty:

    # --------------------------------------------------------
    # The recovered CSV stores the operating window as rows
    # representing minimum / representative / maximum values.
    # --------------------------------------------------------

    window_display = pd.DataFrame(
        columns=[
            "Parameter",
            "Minimum",
            "Representative",
            "Maximum"
        ]
    )

    # Case 1:
    # Three rows containing the three operating conditions.
    if len(operating_window_raw) >= 3:

        row_min = operating_window_raw.iloc[0]
        row_rep = operating_window_raw.iloc[1]
        row_max = operating_window_raw.iloc[2]

        parameter_map = {
            "Agitation (RPM)": "agitation_rpm",
            "Aeration (vvm)": "aeration_vvm",
            "DO (%)": "DO_percent"
        }

        rows = []

        for display_name, column_name in parameter_map.items():

            if column_name in operating_window_raw.columns:

                rows.append(
                    {
                        "Parameter": display_name,
                        "Minimum": row_min[column_name],
                        "Representative": row_rep[column_name],
                        "Maximum": row_max[column_name]
                    }
                )

        window_display = pd.DataFrame(rows)

    if not window_display.empty:

        # Format numbers
        for column in [
            "Minimum",
            "Representative",
            "Maximum"
        ]:

            window_display[column] = pd.to_numeric(
                window_display[column],
                errors="coerce"
            )

        st.dataframe(
            window_display.style.format(
                {
                    "Minimum": "{:.3f}",
                    "Representative": "{:.3f}",
                    "Maximum": "{:.3f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            operating_window_raw,
            use_container_width=True,
            hide_index=True
        )

else:

    st.warning(
        "Operating-window data unavailable."
    )


# ============================================================
# REPRESENTATIVE TARGET-SCALE CONDITION
# ============================================================

st.markdown(
    '<div class="section-title">📍 Representative Target-Scale Condition</div>',
    unsafe_allow_html=True
)

rep_display = {
    "Scale": get_rep_value(
        rep_row,
        ["scale_L"],
        SUPPORTED_SCALE
    ),
    "Working Volume": get_rep_value(
        rep_row,
        ["working_volume_L"]
    ),
    "Tank Diameter": get_rep_value(
        rep_row,
        ["tank_diameter_m"]
    ),
    "Impeller Diameter": get_rep_value(
        rep_row,
        ["impeller_diameter_m"]
    ),
    "Agitation": get_rep_value(
        rep_row,
        ["rpm", "agitation_rpm"]
    ),
    "Aeration": get_rep_value(
        rep_row,
        ["aeration_vvm"]
    ),
    "DO": get_rep_value(
        rep_row,
        ["DO_percent"]
    ),
    "Temperature": get_rep_value(
        rep_row,
        ["temperature_C"]
    ),
    "pH": get_rep_value(
        rep_row,
        ["pH"]
    ),
    "Feed Rate": get_rep_value(
        rep_row,
        ["feed_rate_mL_h"]
    )
}

# Display first four
rep_cols = st.columns(4)

for col, (label, value) in zip(
    rep_cols,
    list(rep_display.items())[:4]
):

    with col:

        st.metric(
            label,
            clean_number(value, 3)
        )


# ============================================================
# BIOLOGICAL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🧫 Predicted Biological Performance</div>',
    unsafe_allow_html=True
)

b1, b2, b3, b4 = st.columns(4)

vcd = first_available(
    biology,
    ["VCD_million_cells_mL", "VCD"],
    get_rep_value(
        rep_row,
        ["VCD_million_cells_mL"],
        0
    )
)

viability = first_available(
    biology,
    ["viability_percent", "Viability"],
    get_rep_value(
        rep_row,
        ["viability_percent"],
        0
    )
)

growth = first_available(
    biology,
    ["growth_rate_per_h", "Growth Rate"],
    get_rep_value(
        rep_row,
        ["growth_rate_per_h"],
        0
    )
)

lactate = first_available(
    biology,
    ["lactate_g_L", "Lactate"],
    get_rep_value(
        rep_row,
        ["lactate_g_L"],
        0
    )
)

with b1:

    st.metric(
        "VCD",
        f"{float(vcd):.3f} M cells/mL"
    )

with b2:

    st.metric(
        "Viability",
        f"{float(viability):.2f}%"
    )

with b3:

    st.metric(
        "Growth Rate",
        f"{float(growth):.4f} /h"
    )

with b4:

    st.metric(
        "Lactate",
        f"{float(lactate):.3f} g/L"
    )


# ============================================================
# ENGINEERING INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">🔬 Scale-Up Engineering Indicators</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Pull directly from representative condition first.
# ------------------------------------------------------------

pv = get_rep_value(
    rep_row,
    ["PV_W_L", "PV_engineered_W_L"]
)

kla = get_rep_value(
    rep_row,
    ["kLa_per_h"]
)

mixing = get_rep_value(
    rep_row,
    ["mixing_time_s"]
)

tip_speed = get_rep_value(
    rep_row,
    ["tip_speed_m_s", "tip_speed_engineered_m_s"]
)

reynolds = get_rep_value(
    rep_row,
    ["reynolds_number", "Re_engineered"]
)

do_value = get_rep_value(
    rep_row,
    ["DO_percent"]
)


# ------------------------------------------------------------
# If values are missing, calculate from geometry/process data.
# ------------------------------------------------------------

rpm = get_rep_value(
    rep_row,
    ["rpm", "agitation_rpm"]
)

impeller_diameter = get_rep_value(
    rep_row,
    ["impeller_diameter_m"]
)

working_volume = get_rep_value(
    rep_row,
    ["working_volume_L"]
)

# Tip speed = pi * D * N
if (
    (tip_speed is None or pd.isna(tip_speed))
    and rpm is not None
    and impeller_diameter is not None
):

    tip_speed = (
        np.pi
        * float(impeller_diameter)
        * float(rpm)
        / 60.0
    )


# Reynolds number = rho * N * D^2 / mu
if (
    (reynolds is None or pd.isna(reynolds))
    and rpm is not None
    and impeller_diameter is not None
):

    rho = 1000.0
    mu = 0.001
    N = float(rpm) / 60.0
    D = float(impeller_diameter)

    reynolds = (
        rho * N * D**2 / mu
    )


# Engineering summary may contain the value if CSV does not.
if pv is None:

    pv = first_available(
        engineering,
        ["PV_W_L", "PV_engineered_W_L"],
        0
    )

if kla is None:

    kla = first_available(
        engineering,
        ["kLa_per_h"],
        0
    )

if mixing is None:

    mixing = first_available(
        engineering,
        ["mixing_time_s"],
        0
    )

if do_value is None:

    do_value = first_available(
        engineering,
        ["DO_percent"],
        0
    )


e1, e2, e3 = st.columns(3)

with e1:

    st.metric(
        "P/V",
        f"{float(pv):.3f} W/L"
        if pv is not None else "--"
    )

with e2:

    st.metric(
        "kLa",
        f"{float(kla):.3f} /h"
        if kla is not None else "--"
    )

with e3:

    st.metric(
        "Mixing Time",
        f"{float(mixing):.1f} s"
        if mixing is not None else "--"
    )


e4, e5, e6 = st.columns(3)

with e4:

    st.metric(
        "Tip Speed",
        f"{float(tip_speed):.3f} m/s"
        if tip_speed is not None else "--"
    )

with e5:

    st.metric(
        "Reynolds Number",
        f"{float(reynolds):,.0f}"
        if reynolds is not None else "--"
    )

with e6:

    st.metric(
        "DO",
        f"{float(do_value):.2f}%"
        if do_value is not None else "--"
    )


# ============================================================
# REAL-TIME RISK GUARDIAN
# ============================================================

st.markdown(
    '<div class="section-title">🛡️ Real-Time Scale-Up Risk Guardian</div>',
    unsafe_allow_html=True
)

guardian_current = pd.DataFrame()

if not guardian_df.empty:

    guardian_working = guardian_df.copy()

    # Parse timestamps
    if "timestamp" in guardian_working.columns:

        guardian_working["timestamp"] = pd.to_datetime(
            guardian_working["timestamp"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the selected/current demo scale instead of simply
    # taking the last row of the entire dataset.
    # --------------------------------------------------------

    if "scale_L" in guardian_working.columns:

        scale_match = guardian_working[
            np.isclose(
                pd.to_numeric(
                    guardian_working["scale_L"],
                    errors="coerce"
                ),
                SUPPORTED_SCALE
            )
        ].copy()

    else:

        scale_match = guardian_working.copy()


    if not scale_match.empty:

        scale_match = scale_match.sort_values(
            "timestamp"
        )

        guardian_current = scale_match.tail(1)


    if not guardian_current.empty:

        latest = guardian_current.iloc[0]

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

        if risk_level.upper() in [
            "HIGH",
            "CRITICAL"
        ]:

            st.warning(
                f"⚠️ **Risk signal:** {risk_reason}"
            )

        else:

            st.info(
                f"ℹ️ **Risk signal:** {risk_reason}"
            )


        st.info(
            f"🔧 **Recommended action:** {recommended_action}"
        )

        st.caption(
            "Risk Guardian output is based on the synthetic/demo "
            "process stream. Model confidence is classification "
            "confidence, not biological certainty."
        )

    else:

        st.info(
            "No Guardian observation available for the current "
            f"{SUPPORTED_SCALE:.0f} L demo scenario."
        )

else:

    st.info(
        "Risk Guardian data unavailable."
    )


# ============================================================
# RISK GUARDIAN TRENDS
# ============================================================

if not guardian_df.empty:

    st.markdown(
        '<div class="section-title">📈 Risk Guardian Trends</div>',
        unsafe_allow_html=True
    )

    trend_df = guardian_df.copy()

    if "scale_L" in trend_df.columns:

        trend_df = trend_df[
            np.isclose(
                pd.to_numeric(
                    trend_df["scale_L"],
                    errors="coerce"
                ),
                SUPPORTED_SCALE
            )
        ].copy()

    if "timestamp" in trend_df.columns:

        trend_df["timestamp"] = pd.to_datetime(
            trend_df["timestamp"],
            errors="coerce"
        )

        trend_df = trend_df.dropna(
            subset=["timestamp"]
        )

        trend_df = trend_df.sort_values(
            "timestamp"
        )

    # --------------------------------------------------------
    # Risk probability chart
    # --------------------------------------------------------

    if (
        not trend_df.empty
        and "failure_probability" in trend_df.columns
    ):

        risk_chart = trend_df[
            ["timestamp", "failure_probability"]
        ].copy()

        risk_chart = risk_chart.set_index(
            "timestamp"
        )

        risk_chart["failure_probability"] *= 100

        st.caption(
            "Failure/deviation probability across the simulated "
            f"{SUPPORTED_SCALE:.0f} L process stream."
        )

        st.line_chart(
            risk_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Process condition chart
    # --------------------------------------------------------

    process_columns = [
        c for c in [
            "DO_percent",
            "oxygen_margin"
        ]
        if c in trend_df.columns
    ]

    if (
        not trend_df.empty
        and process_columns
    ):

        process_chart = trend_df[
            ["timestamp"] + process_columns
        ].copy()

        process_chart = process_chart.set_index(
            "timestamp"
        )

        st.caption(
            "Process-condition trends."
        )

        st.line_chart(
            process_chart,
            use_container_width=True
        )


# ============================================================
# AI COPILOT ASSESSMENT
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Copilot Assessment</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Generate a robust recommendation if JSON structure differs.
# ------------------------------------------------------------

recommendation = first_available(
    ai_recommendation,
    [
        "recommendation",
        "message",
        "summary",
        "copilot_recommendation"
    ],
    None
)

if recommendation is None:

    recommendation = (
        f"Use the representative operating condition as the "
        f"starting point for {SUPPORTED_SCALE:.0f} L target-scale "
        "validation. The operating window should be treated as "
        "a decision-support range and experimentally validated "
        "before production use."
    )


st.markdown(
    f"""
    <div class="copilot-box">

    <b>🤖 AI Copilot Recommendation</b>

    <p>{recommendation}</p>

    <p>
    <b>Target scale:</b> {SUPPORTED_SCALE:.0f} L
    </p>

    <p>
    <b>Process status:</b> {status}
    </p>

    <p>
    <b>Readiness:</b> {float(overall_score):.2f} / 100
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WHY IS THE PROCESS AT RISK?
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Why is the Process at Risk?</div>',
    unsafe_allow_html=True
)

if not guardian_current.empty:

    latest = guardian_current.iloc[0]

    warning_columns = [
        ("DO", "DO_warning"),
        ("Oxygen Transfer", "oxygen_warning"),
        ("Viability", "viability_warning"),
        ("pH", "pH_warning"),
        ("Mixing", "mixing_warning")
    ]

    active_warnings = []

    for label, column in warning_columns:

        if column in guardian_current.columns:

            value = latest[column]

            if bool(value):
                active_warnings.append(label)

    if active_warnings:

        st.warning(
            "**Active warning signals:** "
            + ", ".join(active_warnings)
        )

    else:

        st.success(
            "No active rule-based warning signals detected "
            "in the current displayed Guardian observation."
        )

else:

    st.info(
        "No current Guardian observation available."
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
