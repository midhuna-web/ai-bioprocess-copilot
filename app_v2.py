# ============================================================
# SCALEWISE AI COPILOT
# STEP 6A v2 — FUNCTIONAL DASHBOARD
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ScaleWise AI Copilot",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model_artifacts"


# ============================================================
# TITLE
# ============================================================

st.title("🧬 ScaleWise AI Copilot")

st.caption(
    "AI Copilot for scalable cell-culture bioprocess design"
)

st.info(
    "Prototype demonstration using synthetic/demo data. "
    "Predictions, risk assessments and operating recommendations "
    "require experimental validation."
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_models():

    return joblib.load(
        ARTIFACT_DIR / "model1_final_models.joblib"
    )


@st.cache_resource
def load_preprocessor():

    return joblib.load(
        ARTIFACT_DIR / "preprocessor.joblib"
    )


@st.cache_resource
def load_metadata():

    path = ARTIFACT_DIR / "model1_metadata.joblib"

    if path.exists():
        return joblib.load(path)

    return {}


@st.cache_resource
def load_schema():

    path = ARTIFACT_DIR / "feature_schema.joblib"

    if path.exists():
        return joblib.load(path)

    return {}


@st.cache_data
def load_physics():

    return pd.read_csv(
        ARTIFACT_DIR / "physics_engine.csv"
    )


@st.cache_data
def load_model1_data():

    return pd.read_csv(
        ARTIFACT_DIR / "model1_clean.csv"
    )


# ============================================================
# LOAD
# ============================================================

try:

    model1_models = load_models()
    preprocessor = load_preprocessor()
    metadata = load_metadata()
    feature_schema = load_schema()

    physics_df = load_physics()
    model1_df = load_model1_data()

except Exception as e:

    st.error("ScaleWise artifacts could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# DETERMINE AVAILABLE CATEGORIES
# ============================================================

def find_column_values(df, column, fallback):

    if column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            return sorted(values)

    return fallback


cell_lines = find_column_values(
    model1_df,
    "cell_line",
    [
        "Chicken fibroblast",
        "Trout",
        "Snow trout"
    ]
)


immobilization_values = find_column_values(
    model1_df,
    "immobilization",
    [
        "Not immobilized",
        "Immobilized"
    ]
)


impeller_types = find_column_values(
    model1_df,
    "impeller_type",
    [
        "Rushton",
        "Marine",
        "Pitched blade"
    ]
)


# ============================================================
# SIDEBAR — USER INPUTS
# ============================================================

st.sidebar.header("⚙️ Process Configuration")

st.sidebar.subheader("Biological Parameters")


cell_line = st.sidebar.selectbox(
    "Cell line",
    cell_lines
)


immobilization = st.sidebar.selectbox(
    "Immobilization",
    immobilization_values
)


target_scale = st.sidebar.selectbox(
    "Target scale (L)",
    sorted(
        physics_df["scale_L"]
        .dropna()
        .unique()
        .tolist()
    )
)


st.sidebar.subheader("Process Parameters")


working_time = st.sidebar.number_input(
    "Working time (h)",
    min_value=0.1,
    value=24.0,
    step=1.0
)


rpm = st.sidebar.number_input(
    "Agitation speed (rpm)",
    min_value=1.0,
    value=220.0,
    step=5.0
)


aeration = st.sidebar.number_input(
    "Aeration (vvm)",
    min_value=0.0,
    value=0.06,
    step=0.01,
    format="%.3f"
)


temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=37.0,
    step=0.5
)


pH = st.sidebar.number_input(
    "pH",
    value=7.0,
    step=0.05,
    format="%.2f"
)


DO = st.sidebar.number_input(
    "DO (%)",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=1.0
)


initial_VCD = st.sidebar.number_input(
    "Initial VCD (million cells/mL)",
    min_value=0.0,
    value=0.5,
    step=0.1
)


feed_rate = st.sidebar.number_input(
    "Feed rate (mL/h)",
    min_value=0.0,
    value=0.5,
    step=0.1
)


impeller_type = st.sidebar.selectbox(
    "Impeller type",
    impeller_types
)


run_analysis = st.sidebar.button(
    "🚀 RUN SCALEWISE ANALYSIS",
    use_container_width=True
)


# ============================================================
# TARGET SCALE REFERENCE ENGINE
# ============================================================

def get_reference_row():

    target_data = physics_df[
        physics_df["scale_L"] == target_scale
    ].copy()

    if len(target_data) == 0:

        target_data = physics_df.copy()

    # Find closest operating condition
    target_data["distance"] = (

        abs(target_data["rpm"] - rpm)
        / max(target_data["rpm"].std(), 1)

        +

        abs(target_data["aeration_vvm"] - aeration)
        / max(target_data["aeration_vvm"].std(), 0.001)

        +

        abs(target_data["DO_percent"] - DO)
        / max(target_data["DO_percent"].std(), 1)

        +

        abs(target_data["temperature_C"] - temperature)
        / max(target_data["temperature_C"].std(), 1)

        +

        abs(target_data["pH"] - pH)
        / max(target_data["pH"].std(), 0.1)
    )

    row = target_data.sort_values(
        "distance"
    ).iloc[0].copy()

    return row


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input():

    ref = get_reference_row()

    row = {}

    # --------------------------------------------------------
    # Process variables
    # --------------------------------------------------------

    row["cell_line"] = cell_line
    row["immobilization"] = immobilization
    row["scale_L"] = target_scale
    row["working_time_h"] = working_time
    row["rpm"] = rpm
    row["aeration_vvm"] = aeration
    row["temperature_C"] = temperature
    row["pH"] = pH
    row["DO_percent"] = DO
    row["initial_VCD_million_cells_mL"] = initial_VCD
    row["feed_rate_mL_h"] = feed_rate

    # --------------------------------------------------------
    # Reference physical variables
    # --------------------------------------------------------

    row["PV_V_L"] = ref["PV_V_L"]
    row["kLa_per_h"] = ref["kLa_per_h"]
    row["mixing_time_s"] = ref["mixing_time_s"]
    row["OTR_mmol_L_h"] = ref["OTR_mmol_L_h"]

    row["tank_diameter_mm"] = ref["tank_diameter_mm"]
    row["liquid_height_mm"] = ref["liquid_height_mm"]
    row["impeller_diameter_mm"] = ref["impeller_diameter_mm"]

    row["Reynolds_number"] = ref["Reynolds_number"]
    row["tip_speed_m_s"] = ref["tip_speed_m_s"]

    row["impeller_type"] = impeller_type

    return pd.DataFrame([row])


# ============================================================
# PREDICTION
# ============================================================

def predict_biology(input_df):

    X = input_df.copy()

    # Use exact training feature schema
    if hasattr(preprocessor, "feature_names_in_"):

        expected_features = list(
            preprocessor.feature_names_in_
        )

        for col in expected_features:

            if col not in X.columns:

                X[col] = np.nan

        X = X[expected_features]

    X_transformed = preprocessor.transform(X)

    predictions = {}

    target_order = [
        "VCD_million_cells_mL",
        "viability_percent",
        "growth_rate_per_h",
        "glucose_g_L",
        "lactate_g_L",
        "osmolality_mOsm_kg"
    ]

    for target in target_order:

        if target in model1_models:

            predictions[target] = float(
                model1_models[target]
                .predict(X_transformed)[0]
            )

    return predictions


# ============================================================
# RANGE / COVERAGE ANALYSIS
# ============================================================

def range_score(value, series):

    low = float(series.quantile(0.10))
    high = float(series.quantile(0.90))

    if low <= value <= high:

        return 100.0, "WITHIN OBSERVED REGION"

    elif series.min() <= value <= series.max():

        return 70.0, "WITHIN OBSERVED RANGE"

    else:

        distance = min(
            abs(value - series.min()),
            abs(value - series.max())
        )

        span = max(
            series.max() - series.min(),
            1e-9
        )

        penalty = min(
            60,
            60 * distance / span
        )

        return max(
            0,
            40 - penalty
        ), "EXTRAPOLATION"


# ============================================================
# PROCESS RISK ENGINE
# ============================================================

def calculate_risk(input_df, predictions):

    row = input_df.iloc[0]

    checks = {}

    # --------------------------------------------------------
    # Operating range checks
    # --------------------------------------------------------

    numeric_parameters = [

        "rpm",
        "aeration_vvm",
        "temperature_C",
        "pH",
        "DO_percent",
        "PV_V_L",
        "kLa_per_h",
        "mixing_time_s",
        "OTR_mmol_L_h",
        "Reynolds_number",
        "tip_speed_m_s"
    ]

    scores = []

    for parameter in numeric_parameters:

        if parameter not in physics_df.columns:

            continue

        score, status = range_score(
            row[parameter],
            physics_df[parameter]
        )

        checks[parameter] = {
            "score": score,
            "status": status
        }

        scores.append(score)

    coverage_score = (
        np.mean(scores)
        if scores
        else 50
    )

    # --------------------------------------------------------
    # Oxygen assessment
    # --------------------------------------------------------

    oxygen_flags = []

    if DO < physics_df["DO_percent"].quantile(0.10):

        oxygen_flags.append(
            "DO is below the lower observed region."
        )

    if row["kLa_per_h"] < physics_df["kLa_per_h"].quantile(0.10):

        oxygen_flags.append(
            "kLa is below the lower observed region."
        )

    if (
        row["OTR_mmol_L_h"]
        < physics_df["OTR_mmol_L_h"].quantile(0.10)
    ):

        oxygen_flags.append(
            "OTR is below the lower observed region."
        )

    # --------------------------------------------------------
    # Hydrodynamic assessment
    # --------------------------------------------------------

    hydro_flags = []

    if (
        row["mixing_time_s"]
        > physics_df["mixing_time_s"].quantile(0.90)
    ):

        hydro_flags.append(
            "Mixing time is high relative to observed data."
        )

    if (
        row["tip_speed_m_s"]
        > physics_df["tip_speed_m_s"].quantile(0.90)
    ):

        hydro_flags.append(
            "Tip speed is high relative to observed data."
        )

    # --------------------------------------------------------
    # Biology assessment
    # --------------------------------------------------------

    biology_flags = []

    if predictions.get(
        "VCD_million_cells_mL",
        0
    ) < model1_df[
        "VCD_million_cells_mL"
    ].quantile(0.25):

        biology_flags.append(
            "Predicted VCD is below the lower biological quartile."
        )

    if predictions.get(
        "lactate_g_L",
        0
    ) > model1_df[
        "lactate_g_L"
    ].quantile(0.75):

        biology_flags.append(
            "Predicted lactate is relatively high."
        )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    penalty = (

        len(oxygen_flags) * 12

        +

        len(hydro_flags) * 10

        +

        len(biology_flags) * 8
    )

    overall_score = max(
        0,
        min(
            100,
            coverage_score - penalty
        )
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if overall_score >= 80:

        status = "GREEN — WITHIN OBSERVED REGION"

    elif overall_score >= 60:

        status = "CONDITIONAL — VALIDATE BEFORE SCALE-UP"

    else:

        status = "RED — SIGNIFICANT EXTRAPOLATION / RISK"

    # --------------------------------------------------------
    # Risk band
    # --------------------------------------------------------

    if overall_score >= 80:

        risk_band = "LOW"

    elif overall_score >= 60:

        risk_band = "WATCH"

    elif overall_score >= 40:

        risk_band = "HIGH"

    else:

        risk_band = "CRITICAL"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = min(
        98,
        max(
            50,
            coverage_score
        )
    )

    # --------------------------------------------------------
    # Warning
    # --------------------------------------------------------

    warnings = (
        oxygen_flags
        + hydro_flags
        + biology_flags
    )

    if len(warnings) == 0:

        warning = (
            "No major rule-based warning detected. "
            "Continue monitoring the selected operating region."
        )

    else:

        warning = " ".join(warnings)

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    if oxygen_flags:

        action = (
            "Review DO, kLa and oxygen-transfer conditions "
            "before proceeding."
        )

    elif hydro_flags:

        action = (
            "Review agitation, mixing time and tip speed "
            "and confirm hydrodynamic similarity."
        )

    elif biology_flags:

        action = (
            "Review predicted biological response and "
            "perform a validation experiment."
        )

    else:

        action = (
            "Continue monitoring and experimentally validate "
            "the selected operating region."
        )

    # --------------------------------------------------------
    # Recommended validation
    # --------------------------------------------------------

    validation = (
        "Run a target-scale validation experiment and "
        "compare VCD, viability, lactate and oxygen-transfer "
        "behaviour against the prediction."
    )

    return {
        "score": overall_score,
        "status": status,
        "risk_band": risk_band,
        "confidence": confidence,
        "warning": warning,
        "action": action,
        "validation": validation,
        "checks": checks,
        "oxygen_flags": oxygen_flags,
        "hydro_flags": hydro_flags,
        "biology_flags": biology_flags
    }


# ============================================================
# HEATMAP GENERATOR
# ============================================================

def make_heatmap(
    x_parameter,
    y_parameter,
    input_df
):

    base = input_df.iloc[0].to_dict()

    # --------------------------------------------------------
    # Available data
    # --------------------------------------------------------

    if x_parameter in physics_df.columns:

        x_series = physics_df[
            x_parameter
        ].dropna()

    elif x_parameter in model1_df.columns:

        x_series = model1_df[
            x_parameter
        ].dropna()

    else:

        x_series = pd.Series([0, 1])

    if y_parameter in physics_df.columns:

        y_series = physics_df[
            y_parameter
        ].dropna()

    elif y_parameter in model1_df.columns:

        y_series = model1_df[
            y_parameter
        ].dropna()

    else:

        y_series = pd.Series([0, 1])

    # --------------------------------------------------------
    # 15 x 15 grid
    # --------------------------------------------------------

    x_values = np.linspace(
        x_series.quantile(0.05),
        x_series.quantile(0.95),
        15
    )

    y_values = np.linspace(
        y_series.quantile(0.05),
        y_series.quantile(0.95),
        15
    )

    results = []

    for y in y_values:

        row_values = []

        for x in x_values:

            candidate = base.copy()

            candidate[x_parameter] = float(x)
            candidate[y_parameter] = float(y)

            candidate_df = pd.DataFrame(
                [candidate]
            )

            try:

                pred = predict_biology(
                    candidate_df
                )

                value = pred.get(
                    "VCD_million_cells_mL",
                    np.nan
                )

            except Exception:

                value = np.nan

            row_values.append(value)

        results.append(row_values)

    heatmap = pd.DataFrame(
        results,
        index=np.round(y_values, 3),
        columns=np.round(x_values, 3)
    )

    return heatmap


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_analysis:

    input_df = build_model_input()

    predictions = predict_biology(
        input_df
    )

    risk = calculate_risk(
        input_df,
        predictions
    )

    st.session_state["input_df"] = input_df
    st.session_state["predictions"] = predictions
    st.session_state["risk"] = risk

    st.session_state["analysis_complete"] = True


# ============================================================
# DEFAULT ANALYSIS
# ============================================================

if "analysis_complete" not in st.session_state:

    input_df = build_model_input()

    try:

        predictions = predict_biology(
            input_df
        )

        risk = calculate_risk(
            input_df,
            predictions
        )

        st.session_state["input_df"] = input_df
        st.session_state["predictions"] = predictions
        st.session_state["risk"] = risk

        st.session_state["analysis_complete"] = True

    except Exception as e:

        st.warning(
            "Click RUN SCALEWISE ANALYSIS after "
            "the model artifacts are loaded."
        )

        st.code(str(e))


# ============================================================
# MAIN DASHBOARD
# ============================================================

if "analysis_complete" in st.session_state:

    predictions = st.session_state[
        "predictions"
    ]

    risk = st.session_state[
        "risk"
    ]

    input_df = st.session_state[
        "input_df"
    ]


    # ========================================================
    # TOP SUMMARY
    # ========================================================

    st.header("📊 ScaleWise Analysis Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Target scale",
        f"{target_scale:g} L"
    )

    c2.metric(
        "Process readiness",
        f"{risk['score']:.1f}/100"
    )

    c3.metric(
        "Risk level",
        risk["risk_band"]
    )

    c4.metric(
        "Confidence",
        f"{risk['confidence']:.1f}%"
    )


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🛡️ Process Risk Scorecard",
            "🚨 Scale-Up Risk Guardian",
            "🔥 Heatmap Explorer",
            "🤖 AI Copilot"
        ]
    )


    # ========================================================
    # TAB 1 — SCORECARD
    # ========================================================

    with tab1:

        st.subheader(
            "Process Risk Scorecard"
        )

        st.metric(
            "Overall Process Readiness",
            f"{risk['score']:.1f}/100"
        )

        if risk["score"] >= 80:

            st.success(
                "GREEN — Operating condition is largely "
                "inside the observed data region."
            )

        elif risk["score"] >= 60:

            st.warning(
                "CONDITIONAL — The condition is usable as a "
                "prototype scenario, but validation is required "
                "before claiming scale-up readiness."
            )

        else:

            st.error(
                "RED — Significant extrapolation or process "
                "risk detected."
            )


        st.markdown("### What does the status mean?")

        st.write(
            f"""
            **{risk['status']}**

            The status is calculated from how closely the selected
            operating condition matches the observed experimental
            region, together with oxygen-transfer, hydrodynamic
            and predicted biological indicators.

            **Conditional does NOT mean process failure.**

            It means that the current condition should undergo
            additional validation before being considered suitable
            for scale-up.
            """
        )


        st.markdown(
            "### Parameter coverage"
        )

        coverage_rows = []

        for parameter, values in risk[
            "checks"
        ].items():

            coverage_rows.append(
                {
                    "Parameter": parameter,
                    "Score": round(
                        values["score"],
                        1
                    ),
                    "Status": values["status"]
                }
            )

        coverage_df = pd.DataFrame(
            coverage_rows
        )

        st.dataframe(
            coverage_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown(
            "### Predicted biological performance"
        )

        biology_display = {
            "VCD (million cells/mL)":
                predictions.get(
                    "VCD_million_cells_mL",
                    np.nan
                ),

            "Viability (%)":
                predictions.get(
                    "viability_percent",
                    np.nan
                ),

            "Growth rate (/h)":
                predictions.get(
                    "growth_rate_per_h",
                    np.nan
                ),

            "Lactate (g/L)":
                predictions.get(
                    "lactate_g_L",
                    np.nan
                )
        }

        st.dataframe(
            pd.DataFrame(
                biology_display,
                index=["Prediction"]
            ).T,
            use_container_width=True
        )


    # ========================================================
    # TAB 2 — RISK GUARDIAN
    # ========================================================

    with tab2:

        st.subheader(
            "🚨 Scale-Up Risk Guardian"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Risk level",
            risk["risk_band"]
        )

        r2.metric(
            "Risk confidence",
            f"{risk['confidence']:.1f}%"
        )

        r3.metric(
            "Warning signals",
            str(
                len(
                    risk["oxygen_flags"]
                    + risk["hydro_flags"]
                    + risk["biology_flags"]
                )
            )
        )


        st.markdown(
            "### 🔔 Risk assessment"
        )

        if risk["risk_band"] == "LOW":

            st.success(
                "LOW RISK — No major rule-based warning "
                "has been detected in the selected region."
            )

        elif risk["risk_band"] == "WATCH":

            st.warning(
                "WATCH — The selected condition requires "
                "additional monitoring and validation."
            )

        elif risk["risk_band"] == "HIGH":

            st.error(
                "HIGH RISK — One or more process conditions "
                "are outside the preferred observed region."
            )

        else:

            st.error(
                "CRITICAL — Significant extrapolation or "
                "multiple process warnings detected."
            )


        st.markdown(
            "### ⚠️ Warning"
        )

        st.write(
            risk["warning"]
        )


        st.markdown(
            "### 🔎 Why is this risky?"
        )

        if risk["oxygen_flags"]:

            st.write(
                "**Oxygen-transfer concern:**"
            )

            for item in risk["oxygen_flags"]:

                st.write(
                    f"• {item}"
                )

        if risk["hydro_flags"]:

            st.write(
                "**Hydrodynamic concern:**"
            )

            for item in risk["hydro_flags"]:

                st.write(
                    f"• {item}"
                )

        if risk["biology_flags"]:

            st.write(
                "**Biological concern:**"
            )

            for item in risk["biology_flags"]:

                st.write(
                    f"• {item}"
                )


        st.markdown(
            "### 🛠️ Suggested action"
        )

        st.info(
            risk["action"]
        )


        st.markdown(
            "### 🧪 Recommended validation"
        )

        st.success(
            risk["validation"]
        )


        st.caption(
            "Risk Guardian is a prototype rule/context-based "
            "decision-support layer. It should not be interpreted "
            "as a validated industrial failure probability."
        )


    # ========================================================
    # TAB 3 — HEATMAP
    # ========================================================

    with tab3:

        st.subheader(
            "🔥 Interactive Operating-Region Heatmap"
        )

        st.write(
            "Select any two numerical process or engineering "
            "parameters to explore their effect on predicted VCD."
        )


        heatmap_parameters = [

            "rpm",
            "aeration_vvm",
            "temperature_C",
            "pH",
            "DO_percent",
            "feed_rate_mL_h",
            "PV_V_L",
            "kLa_per_h",
            "mixing_time_s",
            "OTR_mmol_L_h",
            "Reynolds_number",
            "tip_speed_m_s"
        ]


        h1, h2 = st.columns(2)


        with h1:

            x_parameter = st.selectbox(
                "X-axis parameter",
                heatmap_parameters,
                index=0
            )


        with h2:

            default_y = (
                1
                if heatmap_parameters[1]
                != x_parameter
                else 2
            )

            y_parameter = st.selectbox(
                "Y-axis parameter",
                heatmap_parameters,
                index=default_y
            )


        if x_parameter == y_parameter:

            st.warning(
                "Please select two different parameters."
            )

        else:

            with st.spinner(
                "Generating ScaleWise heatmap..."
            ):

                heatmap_data = make_heatmap(
                    x_parameter,
                    y_parameter,
                    input_df
                )


            fig = px.imshow(
                heatmap_data,
                labels={
                    "x":
                        x_parameter,
                    "y":
                        y_parameter,
                    "color":
                        "Predicted VCD"
                },
                aspect="auto",
                origin="lower"
            )

            fig.update_layout(
                title=(
                    f"Predicted VCD: "
                    f"{x_parameter} vs {y_parameter}"
                ),
                height=600
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


            st.caption(
                "Heatmap values represent model-predicted VCD "
                "for the selected parameter combinations while "
                "other inputs are held at the current process "
                "configuration."
            )


            st.dataframe(
                heatmap_data,
                use_container_width=True
            )


    # ========================================================
    # TAB 4 — AI COPILOT
    # ========================================================

    with tab4:

        st.subheader(
            "🤖 ScaleWise AI Copilot"
        )

        st.write(
            "Ask questions about the current scale-up scenario."
        )


        # ----------------------------------------------------
        # Initialize chat
        # ----------------------------------------------------

        if "copilot_messages" not in st.session_state:

            st.session_state.copilot_messages = []


        # ----------------------------------------------------
        # Display previous messages
        # ----------------------------------------------------

        for message in st.session_state[
            "copilot_messages"
        ]:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )


        # ----------------------------------------------------
        # Chat input
        # ----------------------------------------------------

        question = st.chat_input(
            "Ask ScaleWise anything about this process..."
        )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        if question:

            st.session_state[
                "copilot_messages"
            ].append(
                {
                    "role": "user",
                    "content": question
                }
            )


            q = question.lower()


            # ----------------------------------------------
            # Answer engine
            # ----------------------------------------------

            if (
                "risk" in q
                or "danger" in q
                or "warning" in q
            ):

                answer = f"""
### Current Risk Assessment

**Risk level:** {risk['risk_band']}

**Process status:** {risk['status']}

**Confidence:** {risk['confidence']:.1f}%

**Warning:**  
{risk['warning']}

**Suggested action:**  
{risk['action']}

**Recommended validation:**  
{risk['validation']}
"""


            elif (
                "vcd" in q
                or "biology" in q
                or "prediction" in q
            ):

                answer = f"""
### Predicted Biological Performance

- **VCD:** {predictions.get('VCD_million_cells_mL', np.nan):.3f} million cells/mL
- **Viability:** {predictions.get('viability_percent', np.nan):.2f}%
- **Growth rate:** {predictions.get('growth_rate_per_h', np.nan):.4f} /h
- **Lactate:** {predictions.get('lactate_g_L', np.nan):.3f} g/L
"""


            elif (
                "oxygen" in q
                or "kla" in q
                or "otr" in q
                or "do" in q
            ):

                answer = f"""
### Oxygen-Transfer Assessment

- **DO:** {input_df.iloc[0]['DO_percent']:.2f}%
- **kLa:** {input_df.iloc[0]['kLa_per_h']:.3f} /h
- **OTR:** {input_df.iloc[0]['OTR_mmol_L_h']:.3f} mmol/L/h

The Risk Guardian currently reports:

**{len(risk['oxygen_flags'])} oxygen-transfer warning(s).**
"""


            elif (
                "mixing" in q
                or "hydrodynamic" in q
                or "impeller" in q
                or "agitation" in q
            ):

                answer = f"""
### Hydrodynamic Assessment

- **Agitation:** {input_df.iloc[0]['rpm']:.1f} rpm
- **Mixing time:** {input_df.iloc[0]['mixing_time_s']:.2f} s
- **Tip speed:** {input_df.iloc[0]['tip_speed_m_s']:.3f} m/s
- **Reynolds number:** {input_df.iloc[0]['Reynolds_number']:.0f}

Current hydrodynamic warnings:

**{len(risk['hydro_flags'])}**
"""


            elif (
                "status" in q
                or "conditional" in q
                or "readiness" in q
            ):

                answer = f"""
### Process Readiness

**Score:** {risk['score']:.1f}/100

**Status:** {risk['status']}

Conditional means that the current scenario is not
automatically classified as a failure. It means the
selected operating condition requires additional
validation before being considered suitable for
scale-up.
"""


            elif (
                "validate" in q
                or "validation" in q
                or "experiment" in q
            ):

                answer = f"""
### Recommended Validation

{risk['validation']}

The most important current action is:

**{risk['action']}**
"""


            elif (
                "scale" in q
                or "target" in q
            ):

                answer = f"""
### Current Scale-Up Configuration

- **Target scale:** {target_scale:g} L
- **Cell line:** {cell_line}
- **Immobilization:** {immobilization}
- **Agitation:** {rpm:.1f} rpm
- **Aeration:** {aeration:.3f} vvm
- **Temperature:** {temperature:.2f} °C
- **pH:** {pH:.2f}
- **DO:** {DO:.1f}%

Current readiness:

**{risk['score']:.1f}/100 — {risk['status']}**
"""


            else:

                answer = f"""
### ScaleWise Process Summary

The current target is **{target_scale:g} L**.

The model predicts:

- VCD: **{predictions.get('VCD_million_cells_mL', np.nan):.3f} million cells/mL**
- Viability: **{predictions.get('viability_percent', np.nan):.2f}%**
- Growth rate: **{predictions.get('growth_rate_per_h', np.nan):.4f}/h**
- Lactate: **{predictions.get('lactate_g_L', np.nan):.3f} g/L**

Process readiness is:

**{risk['score']:.1f}/100 — {risk['status']}**

You can ask me about **risk, VCD, biology, oxygen transfer,
mixing, hydrodynamics, scale-up, readiness or validation.**
"""


            # ----------------------------------------------
            # Save assistant response
            # ----------------------------------------------

            st.session_state[
                "copilot_messages"
            ].append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ScaleWise is a decision-support prototype. "
    "Final process decisions require experimental and "
    "process-specific validation."
)
