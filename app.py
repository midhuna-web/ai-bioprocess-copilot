mport streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Copilot for Bioprocess Scale-Up",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background-color: #ffffff;
    margin-bottom: 15px;
}

.metric-label {
    font-size: 13px;
    color: #666;
}

.metric-value {
    font-size: 25px;
    font-weight: 700;
}

.copilot-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    border-left: 5px solid #4c78a8;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧬 AI Copilot for Scalable Cell-Culture Bioprocess Design</div>',
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

target_scale = st.sidebar.number_input(
    "Target Scale (L)",
    min_value=1.0,
    max_value=10000.0,
    value=20.0,
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

st.sidebar.warning(
    "Prototype dashboard using synthetic/demo data. "
    "Process limits require experimental validation."
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
    st.metric("Target Scale", f"{target_scale:.0f} L")

with col2:
    st.metric("Overall Readiness", "-- / 100")

with col3:
    st.metric("Status", "READY FOR VALIDATION")


# ============================================================
# READINESS CATEGORIES
# ============================================================

st.markdown(
    '<div class="section-title">📊 Readiness Categories</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Biological Condition", "--")

with c2:
    st.metric("Oxygen Transfer", "--")

with c3:
    st.metric("Hydrodynamic", "--")

with c4:
    st.metric("Process Control", "--")

with c5:
    st.metric("Model Reliability", "--")


# ============================================================
# OPERATING WINDOW
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Recommended Operating Window</div>',
    unsafe_allow_html=True
)

operating_window = {
    "Parameter": [
        "Agitation (RPM)",
        "Aeration (vvm)",
        "DO (%)",
        "Temperature (°C)",
        "pH",
        "Feed Rate (mL/h)"
    ],
    "Minimum": ["--"] * 6,
    "Representative": ["--"] * 6,
    "Maximum": ["--"] * 6
}

st.dataframe(
    operating_window,
    use_container_width=True,
    hide_index=True
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
    st.metric("VCD", "--")

with b2:
    st.metric("Viability", "--")

with b3:
    st.metric("Growth Rate", "--")

with b4:
    st.metric("Lactate", "--")


# ============================================================
# ENGINEERING INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">🔬 Scale-Up Engineering Indicators</div>',
    unsafe_allow_html=True
)

e1, e2, e3 = st.columns(3)

with e1:
    st.metric("P/V", "--")

with e2:
    st.metric("kLa", "--")

with e3:
    st.metric("Mixing Time", "--")

e4, e5, e6 = st.columns(3)

with e4:
    st.metric("Tip Speed", "--")

with e5:
    st.metric("Reynolds Number", "--")

with e6:
    st.metric("DO", "--")


# ============================================================
# REAL-TIME RISK GUARDIAN
# ============================================================

st.markdown(
    '<div class="section-title">🛡️ Real-Time Scale-Up Risk Guardian</div>',
    unsafe_allow_html=True
)

r1, r2, r3 = st.columns(3)

with r1:
    st.metric("Deviation Probability", "--")

with r2:
    st.metric("Risk Level", "STABLE")

with r3:
    st.metric("Model Confidence", "--")


st.info(
    "Risk Guardian continuously evaluates process conditions, "
    "trend signals and scale-up indicators to identify developing deviations."
)


# ============================================================
# AI COPILOT ASSESSMENT
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Copilot Assessment</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="copilot-box">

<b>AI Copilot Recommendation</b>

<br><br>

The recommended operating region will be generated from the
scale-up optimization and process-readiness analysis.

<br><br>

The system will identify acceptable operating conditions,
predicted biological performance, engineering constraints,
and potential scale-up risks.

</div>
""", unsafe_allow_html=True)


# ============================================================
# RISK EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Why is the Process at Risk?</div>',
    unsafe_allow_html=True
)

st.write(
    "Risk-driver explanations will appear here after connecting "
    "the Model 3 Risk Guardian outputs."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Copilot for Scalable Cell-Culture Bioprocess Design | "
    "Prototype / Synthetic Data Demonstration"
)
