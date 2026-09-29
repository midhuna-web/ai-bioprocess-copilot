
import os
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# ScaleWise — Steps 5A onward
# ============================================================
# Input workbook:
#   Final Data for Hackathon.xlsx
#
# This app contains:
#   5A  Physics / scale-up engine
#   5B  Target-scale interpolation
#   5C  Model-1 predictions
#   5D  Process risk scorecard + scale-up risk assessment
#   5E  Interactive AI Copilot
#
# IMPORTANT:
# The supplied workbook contains failure_event = 0 for every row.
# Therefore no supervised failure classifier is trained. The
# "failure risk" shown by this app is an auditable physics/data-
# coverage/anomaly risk score, NOT a historical failure probability.
# ============================================================


st.set_page_config(
    page_title="ScaleWise — Bioprocess Scale-Up Copilot",
    page_icon="🧬",
    layout="wide",
)


# -----------------------------
# Configuration
# -----------------------------
DATA_FILE = Path(
    os.getenv(
        "SCALewise_DATA_FILE",
        Path(__file__).with_name("Final Data for Hackathon.xlsx")
    )
)

MODEL1_TARGETS = [
    "VCD_million_cells_mL",
    "viability_percent",
    "growth_rate_per_h",
    "glucose_g_L",
    "lactate_g_L",
]

MODEL1_PROCESS_FEATURES = [
    "cell_line",
    "scale_L",
    "batch_age_h",
    "rpm",
    "aeration_vvm",
    "temperature_C",
    "pH",
    "DO_percent",
    "initial_VCD_million_cells_mL",
    "feed_rate_mL_h",
]

MODEL1_PHYSICS_FEATURES = [
    "PV_W_L",
    "kLa_per_h",
    "mixing_time_s",
    "OTR_mmol_L_h",
    "tank_diameter_m",
    "tank_height_m",
    "impeller_diameter_m",
    "reynolds_number",
    "tip_speed_m_s",
    "impeller_type",
]

MODEL1_FEATURES = MODEL1_PROCESS_FEATURES + MODEL1_PHYSICS_FEATURES

PHYSICS_COLUMNS = [
    "scale_L",
    "rpm",
    "aeration_vvm",
    "temperature_C",
    "pH",
    "DO_percent",
    "feed_rate_mL_h",
    "tank_diameter_m",
    "tank_height_m",
    "impeller_diameter_m",
    "impeller_type",
    "PV_W_L",
    "kLa_per_h",
    "mixing_time_s",
    "OTR_mmol_L_h",
    "reynolds_number",
    "tip_speed_m_s",
]

# Conservative operational bounds for the UI.
# Continuous ranges are also checked against the actual workbook.
INPUT_COLUMNS = [
    "rpm",
    "aeration_vvm",
    "temperature_C",
    "pH",
    "DO_percent",
    "feed_rate_mL_h",
    "initial_VCD_million_cells_mL",
    "batch_age_h",
]


# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def pct(x):
    return f"{float(x):.1f}%"


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data(path_string):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Put the Excel file beside app.py "
            "or set SCALewise_DATA_FILE."
        )

    data = pd.read_excel(path)

    required = set(
        MODEL1_TARGETS
        + MODEL1_FEATURES
        + [
            "batch_id",
            "failure_event",
            "medium",
            "serum_condition",
            "process_stage",
        ]
    )
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")

    return data


# -----------------------------
# Step 5A — physics engine
# -----------------------------
@st.cache_data
def build_scale_physics_summary(data):
    numeric = [
        "rpm",
        "aeration_vvm",
        "temperature_C",
        "pH",
        "DO_percent",
        "feed_rate_mL_h",
        "tank_diameter_m",
        "tank_height_m",
        "impeller_diameter_m",
        "PV_W_L",
        "kLa_per_h",
        "mixing_time_s",
        "OTR_mmol_L_h",
        "reynolds_number",
        "tip_speed_m_s",
    ]

    summary = data.groupby("scale_L")[numeric].median().reset_index()
    summary["impeller_type"] = (
        data.groupby("scale_L")["impeller_type"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "Pitched-blade")
        .values
    )
    return summary.sort_values("scale_L").reset_index(drop=True)


def get_neighbors(summary, target_scale):
    scales = summary["scale_L"].to_numpy(dtype=float)

    if target_scale in scales:
        row = summary.loc[summary["scale_L"] == target_scale].iloc[0].to_dict()
        return row, row, True

    if target_scale < scales.min() or target_scale > scales.max():
        raise ValueError(
            f"Target scale must be between {scales.min():g} and {scales.max():g} L."
        )

    lower_scale = scales[scales < target_scale].max()
    upper_scale = scales[scales > target_scale].min()

    lower = summary.loc[summary["scale_L"] == lower_scale].iloc[0].to_dict()
    upper = summary.loc[summary["scale_L"] == upper_scale].iloc[0].to_dict()

    return lower, upper, False


def log_interp(x, x0, x1, y0, y1):
    if x0 == x1:
        return float(y0)
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return float(y0 + w * (y1 - y0))


def target_scale_physics(summary, data, target_scale, rpm, aeration_vvm, temperature_C,
                         pH, DO_percent, feed_rate_mL_h):

    lower, upper, exact = get_neighbors(summary, float(target_scale))

    ref = {}
    for col in [
        "tank_diameter_m",
        "tank_height_m",
        "impeller_diameter_m",
        "PV_W_L",
        "kLa_per_h",
        "mixing_time_s",
        "OTR_mmol_L_h",
        "reynolds_number",
    ]:
        if exact:
            ref[col] = float(lower[col])
        else:
            ref[col] = log_interp(
                target_scale,
                lower["scale_L"],
                upper["scale_L"],
                lower[col],
                upper[col],
            )

    ref_rpm = log_interp(
        target_scale,
        lower["scale_L"],
        upper["scale_L"],
        lower["rpm"],
        upper["rpm"],
    )
    ref_aeration = log_interp(
        target_scale,
        lower["scale_L"],
        upper["scale_L"],
        lower["aeration_vvm"],
        upper["aeration_vvm"],
    )
    ref_do = log_interp(
        target_scale,
        lower["scale_L"],
        upper["scale_L"],
        lower["DO_percent"],
        upper["DO_percent"],
    )

    # Constant-geometry power-density scaling:
    # P/V approximately scales with N^3.
    rpm_ratio = max(float(rpm), 1e-6) / max(ref_rpm, 1e-6)
    pv = ref["PV_W_L"] * rpm_ratio**3

    # Approximate kLa dependence on P/V and gas flow.
    aeration_ratio = max(float(aeration_vvm), 1e-6) / max(ref_aeration, 1e-6)
    kla = ref["kLa_per_h"] * math.sqrt(max(pv, 1e-9) / max(ref["PV_W_L"], 1e-9))
    kla *= math.sqrt(aeration_ratio)

    # Mixing time: approximately inversely related to rpm,
    # with scale dependence already represented in the reference.
    mixing = ref["mixing_time_s"] / max(rpm_ratio, 1e-6)

    # Reynolds number scales approximately linearly with rpm
    # when fluid properties and geometry are held at the target-scale reference.
    reynolds = ref["reynolds_number"] * rpm_ratio

    # Independent tip-speed calculation.
    tip_speed = math.pi * ref["impeller_diameter_m"] * float(rpm) / 60.0

    # OTR approximation using kLa and DO relative to reference.
    otr = ref["OTR_mmol_L_h"]
    otr *= max(kla, 1e-9) / max(ref["kLa_per_h"], 1e-9)
    otr *= max(float(DO_percent), 1.0) / max(ref_do, 1.0)

    impeller_type = (
        lower["impeller_type"] if exact or lower["impeller_type"] == upper["impeller_type"]
        else lower["impeller_type"]
    )

    result = {
        "scale_L": float(target_scale),
        "rpm": float(rpm),
        "aeration_vvm": float(aeration_vvm),
        "temperature_C": float(temperature_C),
        "pH": float(pH),
        "DO_percent": float(DO_percent),
        "feed_rate_mL_h": float(feed_rate_mL_h),
        "tank_diameter_m": float(ref["tank_diameter_m"]),
        "tank_height_m": float(ref["tank_height_m"]),
        "impeller_diameter_m": float(ref["impeller_diameter_m"]),
        "impeller_type": str(impeller_type),
        "PV_W_L": float(pv),
        "kLa_per_h": float(kla),
        "mixing_time_s": float(mixing),
        "OTR_mmol_L_h": float(otr),
        "reynolds_number": float(reynolds),
        "tip_speed_m_s": float(tip_speed),
        "lower_reference_scale_L": float(lower["scale_L"]),
        "upper_reference_scale_L": float(upper["scale_L"]),
        "reference_rpm": float(ref_rpm),
        "reference_aeration_vvm": float(ref_aeration),
        "reference_DO_percent": float(ref_do),
    }

    return result


# -----------------------------
# Step 3A/3B/3C — model training
# -----------------------------
@st.cache_resource
def train_model1(data):
    # Batch-aware split.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(
        splitter.split(data, groups=data["batch_id"])
    )

    train_val = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    tr_idx, val_idx = next(
        splitter2.split(train_val, groups=train_val["batch_id"])
    )

    train = train_val.iloc[tr_idx].copy()
    val = train_val.iloc[val_idx].copy()

    X_train = train[MODEL1_FEATURES].copy()
    X_val = val[MODEL1_FEATURES].copy()
    X_test = test[MODEL1_FEATURES].copy()

    cat_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    num_cols = [c for c in MODEL1_FEATURES if c not in cat_cols]

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        [
            ("cat", encoder, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    Xtr = preprocessor.fit_transform(X_train)
    Xv = preprocessor.transform(X_val)
    Xt = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    best_models = {}
    validation_rows = []
    test_rows = []

    for target in MODEL1_TARGETS:
        ytr = train[target].to_numpy()
        yv = val[target].to_numpy()
        yt = test[target].to_numpy()

        candidates = {
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=2,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42,
            ),
        }

        fitted = {}
        for name, model in candidates.items():
            model.fit(Xtr, ytr)
            pred_v = model.predict(Xv)

            validation_rows.append({
                "target": target,
                "model": name,
                "R2": r2_score(yv, pred_v),
                "MAE": mean_absolute_error(yv, pred_v),
                "RMSE": rmse(yv, pred_v),
            })
            fitted[name] = model

        # Select using validation only.
        target_val = [
            r for r in validation_rows if r["target"] == target
        ]
        selected_name = max(target_val, key=lambda r: r["R2"])["model"]
        selected = fitted[selected_name]
        best_models[target] = selected

        pred_t = selected.predict(Xt)
        test_rows.append({
            "target": target,
            "selected_model": selected_name,
            "R2": r2_score(yt, pred_t),
            "MAE": mean_absolute_error(yt, pred_t),
            "RMSE": rmse(yt, pred_t),
        })

    # Anomaly model for risk assessment.
    # It is deliberately not called a failure classifier.
    anomaly = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    anomaly.fit(Xtr)

    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "best_models": best_models,
        "validation_results": pd.DataFrame(validation_rows),
        "test_results": pd.DataFrame(test_rows),
        "anomaly_model": anomaly,
        "train": train,
        "validation": val,
        "test": test,
    }


# -----------------------------
# Step 5C — build model input
# -----------------------------
def build_model_input(
    physics,
    cell_line,
    batch_age_h,
    initial_VCD_million_cells_mL,
):
    row = {
        "cell_line": cell_line,
        "scale_L": physics["scale_L"],
        "batch_age_h": batch_age_h,
        "rpm": physics["rpm"],
        "aeration_vvm": physics["aeration_vvm"],
        "temperature_C": physics["temperature_C"],
        "pH": physics["pH"],
        "DO_percent": physics["DO_percent"],
        "initial_VCD_million_cells_mL": initial_VCD_million_cells_mL,
        "feed_rate_mL_h": physics["feed_rate_mL_h"],
        "PV_W_L": physics["PV_W_L"],
        "kLa_per_h": physics["kLa_per_h"],
        "mixing_time_s": physics["mixing_time_s"],
        "OTR_mmol_L_h": physics["OTR_mmol_L_h"],
        "tank_diameter_m": physics["tank_diameter_m"],
        "tank_height_m": physics["tank_height_m"],
        "impeller_diameter_m": physics["impeller_diameter_m"],
        "reynolds_number": physics["reynolds_number"],
        "tip_speed_m_s": physics["tip_speed_m_s"],
        "impeller_type": physics["impeller_type"],
    }

    return pd.DataFrame([row])[MODEL1_FEATURES]


def predict_model1(model_bundle, model_input):
    X = model_bundle["preprocessor"].transform(model_input)
    predictions = {}

    for target, model in model_bundle["best_models"].items():
        predictions[target] = float(model.predict(X)[0])

    return pd.DataFrame([predictions])


# -----------------------------
# Step 5D — risk engine
# -----------------------------
def percentile_distance(value, low, high):
    if value < low:
        return (low - value) / max(abs(high - low), 1e-9)
    if value > high:
        return (value - high) / max(abs(high - low), 1e-9)
    return 0.0


def build_risk_score(
    data,
    model_bundle,
    physics,
    model_input,
    target_scale,
):
    # Use the central 90% of observed data as a "normal operating envelope".
    risk_specs = {
        "rpm": ("rpm", "RPM"),
        "aeration_vvm": ("aeration_vvm", "Aeration"),
        "DO_percent": ("DO_percent", "Dissolved oxygen"),
        "PV_W_L": ("PV_W_L", "Power density"),
        "kLa_per_h": ("kLa_per_h", "kLa"),
        "mixing_time_s": ("mixing_time_s", "Mixing time"),
        "OTR_mmol_L_h": ("OTR_mmol_L_h", "OTR"),
        "reynolds_number": ("reynolds_number", "Reynolds number"),
        "tip_speed_m_s": ("tip_speed_m_s", "Tip speed"),
    }

    factors = []

    for key, (column, label) in risk_specs.items():
        q05, q95 = data[column].quantile([0.05, 0.95])
        value = float(physics[key])
        dist = percentile_distance(value, q05, q95)

        if dist > 0:
            # Convert envelope excursion to 0–30 factor points.
            score = min(30.0, 12.0 + 30.0 * dist)
            factors.append({
                "factor": label,
                "value": value,
                "reference_low": q05,
                "reference_high": q95,
                "factor_score": score,
                "message": (
                    f"{label}={value:.3g} is outside the central 90% "
                    f"training envelope ({q05:.3g}–{q95:.3g})."
                ),
            })

    # Scale interpolation / extrapolation risk.
    observed_scales = sorted(data["scale_L"].unique().tolist())
    min_scale = min(observed_scales)
    max_scale = max(observed_scales)

    if target_scale < min_scale or target_scale > max_scale:
        factors.append({
            "factor": "Scale extrapolation",
            "value": target_scale,
            "reference_low": min_scale,
            "reference_high": max_scale,
            "factor_score": 35.0,
            "message": "Target scale is outside the observed dataset range.",
        })
    elif target_scale not in observed_scales:
        lower = max(s for s in observed_scales if s < target_scale)
        upper = min(s for s in observed_scales if s > target_scale)
        gap = math.log(upper / lower)
        factors.append({
            "factor": "Scale interpolation",
            "value": target_scale,
            "reference_low": lower,
            "reference_high": upper,
            "factor_score": min(18.0, 5.0 + 5.0 * gap),
            "message": (
                f"Target scale is interpolated between {lower:g} L and {upper:g} L."
            ),
        })

    # Model anomaly score.
    X = model_bundle["preprocessor"].transform(model_input)
    raw_anomaly = float(model_bundle["anomaly_model"].decision_function(X)[0])
    is_anomaly = int(model_bundle["anomaly_model"].predict(X)[0]) == -1

    if is_anomaly:
        factors.append({
            "factor": "Multivariate operating envelope",
            "value": raw_anomaly,
            "reference_low": None,
            "reference_high": None,
            "factor_score": 25.0,
            "message": (
                "The combined input/physics vector is unusual relative to "
                "the training distribution."
            ),
        })

    # Keep total in 0–100.
    raw_score = sum(f["factor_score"] for f in factors)
    risk_score = float(min(100.0, raw_score))

    if risk_score < 25:
        level = "LOW"
    elif risk_score < 50:
        level = "MODERATE"
    elif risk_score < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    # Evidence confidence, not probability of failure.
    #
    # High when:
    #   - target is inside observed scale range
    #   - target is close to observed reference scales
    #   - input is inside central training envelope
    #
    # Lower when:
    #   - target is interpolated over a large scale gap
    #   - physics are outside training envelope
    #   - multivariate anomaly is detected
    confidence = 92.0

    if target_scale not in observed_scales:
        confidence -= 8.0

    if any(f["factor"] == "Multivariate operating envelope" for f in factors):
        confidence -= 18.0

    out_of_envelope = sum(
        1 for f in factors
        if f["factor"] not in ["Scale interpolation", "Scale extrapolation",
                               "Multivariate operating envelope"]
    )
    confidence -= min(35.0, 6.0 * out_of_envelope)

    confidence = float(np.clip(confidence, 35.0, 95.0))

    # Prioritize largest factor.
    factors_sorted = sorted(
        factors, key=lambda x: x["factor_score"], reverse=True
    )

    if not factors_sorted:
        suggested_action = (
            "Proceed to engineering verification; no major data-envelope "
            "warning was detected."
        )
        validation = [
            "Confirm target-scale geometry and impeller configuration.",
            "Run mixing-time and kLa verification at pilot scale.",
            "Confirm oxygen-transfer capacity before production use.",
        ]
    else:
        top = factors_sorted[0]["factor"]
        if "oxygen" in top.lower() or top == "OTR":
            suggested_action = (
                "Verify oxygen-transfer capacity and DO control before increasing scale."
            )
            validation = [
                "Measure kLa at target operating conditions.",
                "Run an OTR/OUR balance or oxygen-transfer challenge test.",
                "Confirm DO cascade response and alarm limits.",
            ]
        elif "mixing" in top.lower():
            suggested_action = (
                "Verify mixing performance before scale-up; do not rely on geometry alone."
            )
            validation = [
                "Measure mixing time using a tracer or equivalent method.",
                "Check local pH/DO gradients.",
                "Verify impeller clearance and power draw.",
            ]
        elif "power" in top.lower() or "RPM" in top:
            suggested_action = (
                "Check mechanical power input and shear exposure before scale-up."
            )
            validation = [
                "Verify P/V and shaft power experimentally.",
                "Check tip speed and shear-sensitive cell response.",
                "Confirm motor/drive operating margin.",
            ]
        elif "Scale" in top:
            suggested_action = (
                "Treat this as a scale-interpolation uncertainty and add an intermediate validation scale."
            )
            validation = [
                "Validate at an intermediate scale.",
                "Compare mixing, kLa, P/V and DO response against the model.",
                "Use the measured pilot data to recalibrate the scale-up model.",
            ]
        else:
            suggested_action = (
                "Hold the proposed condition for engineering review and targeted pilot validation."
            )
            validation = [
                "Repeat the prediction with measured target-scale physics.",
                "Run a controlled pilot batch.",
                "Compare measured process trajectories with model predictions.",
            ]

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "confidence": confidence,
        "factors": factors_sorted,
        "suggested_action": suggested_action,
        "recommended_validation": validation,
        "anomaly_score": raw_anomaly,
        "failure_model_available": False,
    }


# -----------------------------
# AI Copilot
# -----------------------------
def copilot_answer(user_message, context, chat_history):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "AI Copilot is not connected because OPENAI_API_KEY is not set. "
            "The process-risk and prediction modules are still available. "
            "Set OPENAI_API_KEY in the deployment secrets/environment to enable live AI chat."
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

        system_instructions = """
You are ScaleWise Copilot, an engineering decision-support assistant for
bioprocess/fermentation scale-up.

Rules:
1. Use the supplied current run context as the primary source.
2. Do not invent experimental results.
3. Distinguish model prediction, physics estimate, risk score and confirmed
   experimental evidence.
4. The workbook's failure_event column contains only zeros, so there is no
   supervised historical failure probability. Never present the risk score
   as a calibrated probability of batch failure.
5. Explain why a warning exists and give practical validation steps.
6. If the user asks for a change in an input, explain the expected direction
   of effect but recommend validation before implementation.
7. Be concise but technically useful.
"""

        conversation = []
        for item in chat_history[-8:]:
            conversation.append({
                "role": item["role"],
                "content": item["content"],
            })

        prompt = (
            f"{system_instructions}\n\n"
            f"CURRENT SCALEWISE CONTEXT:\n{json.dumps(context, indent=2, default=str)}\n\n"
            f"USER QUESTION:\n{user_message}"
        )

        response = client.responses.create(
            model=model_name,
            input=prompt,
        )

        return response.output_text

    except Exception as exc:
        return (
            "The AI Copilot could not complete the request. "
            f"Technical message: {exc}"
        )


# ============================================================
# UI
# ============================================================
st.title("🧬 ScaleWise")
st.caption(
    "Bioprocess scale-up prediction • physics-based risk assessment • AI copilot"
)

with st.sidebar:
    st.header("1. Define the process")

    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        st.stop()

    try:
        data = load_data(str(DATA_FILE))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    scale_summary = build_scale_physics_summary(data)

    observed_scales = sorted(data["scale_L"].unique().tolist())

    # Include 50 L because it is a useful interpolated scale, while
    # keeping observed scales visibly distinct.
    suggested_interpolated = [50, 250, 750, 2500, 7500]
    scale_options = sorted(
        set(
            [float(x) for x in observed_scales]
            + [
                float(x)
                for x in suggested_interpolated
                if min(observed_scales) <= x <= max(observed_scales)
            ]
        )
    )

    target_scale = st.selectbox(
        "Target working volume (L)",
        scale_options,
        index=scale_options.index(50.0) if 50.0 in scale_options else 0,
        help="Observed scales come from the workbook. Additional values are interpolated."
    )

    cell_line = st.selectbox(
        "Cell line",
        sorted(data["cell_line"].dropna().unique().tolist())
    )

    impeller_type = st.selectbox(
        "Impeller type",
        sorted(data["impeller_type"].dropna().unique().tolist())
    )

    medium = st.selectbox(
        "Medium",
        sorted(data["medium"].dropna().unique().tolist())
    )

    serum_condition = st.selectbox(
        "Serum condition",
        sorted(data["serum_condition"].dropna().unique().tolist())
    )

    process_stage = st.selectbox(
        "Process stage",
        sorted(data["process_stage"].dropna().unique().tolist())
    )

    def slider_from_data(column, label, step):
        lo = float(data[column].min())
        hi = float(data[column].max())
        default = float(data[column].median())
        return st.slider(
            label,
            min_value=lo,
            max_value=hi,
            value=default,
            step=step,
        )

    rpm = slider_from_data("rpm", "Agitation (rpm)", 1.0)
    aeration = slider_from_data("aeration_vvm", "Aeration (vvm)", 0.005)
    temperature = slider_from_data("temperature_C", "Temperature (°C)", 0.1)
    pH = slider_from_data("pH", "pH", 0.01)
    DO = slider_from_data("DO_percent", "DO (%)", 0.5)
    feed = slider_from_data("feed_rate_mL_h", "Feed rate (mL/h)", 0.01)
    initial_vcd = slider_from_data(
        "initial_VCD_million_cells_mL",
        "Initial VCD (million cells/mL)",
        0.005,
    )
    batch_age = slider_from_data("batch_age_h", "Batch age (h)", 0.5)

    st.divider()
    st.caption(
        f"Dataset: {len(data):,} rows • observed scales: "
        + ", ".join(f"{int(x):,}" for x in observed_scales)
        + " L"
    )

    if data["failure_event"].nunique() == 1:
        st.warning(
            "Failure labels are not usable: failure_event has only one class (0). "
            "Risk is therefore physics/data-envelope based."
        )


# Train/cache models.
with st.spinner("Preparing physics and prediction models…"):
    model_bundle = train_model1(data)


# -----------------------------
# Current prediction
# -----------------------------
physics = target_scale_physics(
    scale_summary,
    data,
    target_scale=float(target_scale),
    rpm=rpm,
    aeration_vvm=aeration,
    temperature_C=temperature,
    pH=pH,
    DO_percent=DO,
    feed_rate_mL_h=feed,
)

model_input = build_model_input(
    physics,
    cell_line=cell_line,
    batch_age_h=batch_age,
    initial_VCD_million_cells_mL=initial_vcd,
)

predictions = predict_model1(model_bundle, model_input)

risk = build_risk_score(
    data,
    model_bundle,
    physics,
    model_input,
    target_scale=float(target_scale),
)


# ============================================================
# Dashboard
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "⚙️ Scale-Up Physics",
        "🚦 Risk Scorecard",
        "🧪 Validation",
        "🤖 AI Copilot",
    ]
)

with tab1:
    st.subheader("Scale-up decision dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target scale", f"{target_scale:,.0f} L")
    c2.metric("Risk score", f"{risk['risk_score']:.0f}/100")
    c3.metric("Risk level", risk["risk_level"])
    c4.metric("Evidence confidence", f"{risk['confidence']:.0f}%")

    st.divider()

    st.markdown("### Predicted biological outcomes")

    prediction_display = predictions.T.reset_index()
    prediction_display.columns = ["Variable", "Predicted value"]
    st.dataframe(
        prediction_display,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Current engineering condition")

    eng_cols = [
        "scale_L",
        "rpm",
        "aeration_vvm",
        "DO_percent",
        "temperature_C",
        "pH",
        "tank_diameter_m",
        "tank_height_m",
        "impeller_diameter_m",
        "PV_W_L",
        "kLa_per_h",
        "mixing_time_s",
        "OTR_mmol_L_h",
        "reynolds_number",
        "tip_speed_m_s",
        "impeller_type",
    ]
    st.dataframe(
        pd.DataFrame([physics])[eng_cols].T.rename(columns={0: "Value"}),
        use_container_width=True,
    )

    st.info(
        "The dashboard combines Model-1 biological predictions with a "
        "physics/data-envelope risk assessment. It is a decision-support "
        "tool, not a substitute for pilot validation."
    )


with tab2:
    st.subheader("5A–5B: Target-scale physics engine")

    st.markdown(
        "The target scale is interpolated between the nearest observed scales "
        "when it is not directly present in the workbook."
    )

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**Reference scales**")
        st.write(
            f"Lower reference: **{physics['lower_reference_scale_L']:g} L**"
        )
        st.write(
            f"Upper reference: **{physics['upper_reference_scale_L']:g} L**"
        )

    with p2:
        st.markdown("**Reference operating point**")
        st.write(f"Reference rpm: **{physics['reference_rpm']:.2f}**")
        st.write(
            f"Reference aeration: **{physics['reference_aeration_vvm']:.4f} vvm**"
        )
        st.write(
            f"Reference DO: **{physics['reference_DO_percent']:.2f}%**"
        )

    st.markdown("### Calculated scale-up physics")

    physics_table = pd.DataFrame({
        "Parameter": [
            "Tank diameter",
            "Tank height",
            "Impeller diameter",
            "P/V",
            "kLa",
            "Mixing time",
            "OTR",
            "Reynolds number",
            "Tip speed",
            "Impeller type",
        ],
        "Value": [
            f"{physics['tank_diameter_m']:.4f} m",
            f"{physics['tank_height_m']:.4f} m",
            f"{physics['impeller_diameter_m']:.4f} m",
            f"{physics['PV_W_L']:.5f} W/L",
            f"{physics['kLa_per_h']:.3f} 1/h",
            f"{physics['mixing_time_s']:.2f} s",
            f"{physics['OTR_mmol_L_h']:.3f} mmol/L/h",
            f"{physics['reynolds_number']:.1f}",
            f"{physics['tip_speed_m_s']:.3f} m/s",
            physics["impeller_type"],
        ],
    })
    st.dataframe(physics_table, use_container_width=True, hide_index=True)

    st.markdown("### Scale reference table")
    st.dataframe(
        scale_summary[
            [
                "scale_L",
                "rpm",
                "aeration_vvm",
                "PV_W_L",
                "kLa_per_h",
                "mixing_time_s",
                "OTR_mmol_L_h",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


with tab3:
    st.subheader("Process risk scorecard")

    level = risk["risk_level"]

    if level == "LOW":
        st.success(f"Risk: {level} — score {risk['risk_score']:.0f}/100")
    elif level == "MODERATE":
        st.warning(f"Risk: {level} — score {risk['risk_score']:.0f}/100")
    else:
        st.error(f"Risk: {level} — score {risk['risk_score']:.0f}/100")

    st.metric(
        "Assessment confidence",
        f"{risk['confidence']:.0f}%",
        help=(
            "This is evidence/coverage confidence in the risk assessment. "
            "It is NOT the probability of batch failure."
        ),
    )

    st.markdown("### Scale-up failure predictor")

    st.write(
        f"**Assessment:** {risk['risk_level']} risk of scale-up difficulty"
    )
    st.write(
        f"**Confidence:** {risk['confidence']:.0f}% evidence/coverage confidence"
    )

    st.caption(
        "Important: the uploaded data contains no positive failure events. "
        "Therefore this assessment is not a calibrated probability of failure."
    )

    st.markdown("### Warning factors")

    if not risk["factors"]:
        st.success("No major warning factors detected.")
    else:
        rows = []
        for f in risk["factors"]:
            rows.append({
                "Warning": f["factor"],
                "Value": f"{f['value']:.4g}",
                "Reference": (
                    "—"
                    if f["reference_low"] is None
                    else f"{f['reference_low']:.4g} – {f['reference_high']:.4g}"
                ),
                "Risk contribution": f"{f['factor_score']:.1f}",
                "Interpretation": f["message"],
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Suggested action")
    st.info(risk["suggested_action"])

    st.markdown("### Recommended validation")
    for item in risk["recommended_validation"]:
        st.write("• " + item)


with tab4:
    st.subheader("Recommended validation plan")

    st.markdown("### Model performance")

    st.dataframe(
        model_bundle["test_results"].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Target-scale validation sequence")

    validation_steps = [
        ("1. Geometry", "Verify tank diameter, liquid height, impeller diameter, clearance and working volume."),
        ("2. Hydrodynamics", "Measure/verify P/V, tip speed, Reynolds number and mixing time."),
        ("3. Oxygen transfer", "Measure kLa and confirm OTR under the proposed aeration/DO condition."),
        ("4. Control response", "Challenge the DO/pH control loop and confirm actuator capacity."),
        ("5. Biological pilot", "Run a controlled pilot batch and compare VCD, viability, glucose and lactate trajectories."),
        ("6. Model update", "Use measured pilot data to recalibrate the scale-up model before the next scale."),
    ]

    for title, description in validation_steps:
        st.markdown(f"**{title}** — {description}")

    st.markdown("### Current target-scale input")
    st.dataframe(
        model_input.T.rename(columns={0: "Value"}),
        use_container_width=True,
    )


with tab5:
    st.subheader("🤖 Interactive ScaleWise AI Copilot")

    st.caption(
        "Ask about the current scale-up scenario, warnings, predicted outcomes, "
        "or what validation should be performed next."
    )

    context = {
        "target_scale_L": target_scale,
        "cell_line": cell_line,
        "medium": medium,
        "serum_condition": serum_condition,
        "process_stage": process_stage,
        "user_inputs": {
            "rpm": rpm,
            "aeration_vvm": aeration,
            "temperature_C": temperature,
            "pH": pH,
            "DO_percent": DO,
            "feed_rate_mL_h": feed,
            "initial_VCD_million_cells_mL": initial_vcd,
            "batch_age_h": batch_age,
            "impeller_type_requested": impeller_type,
        },
        "physics": physics,
        "predictions": predictions.iloc[0].to_dict(),
        "risk": {
            "score": risk["risk_score"],
            "level": risk["risk_level"],
            "confidence": risk["confidence"],
            "warnings": [f["message"] for f in risk["factors"]],
            "suggested_action": risk["suggested_action"],
            "recommended_validation": risk["recommended_validation"],
        },
        "data_note": (
            "failure_event has only one class (0) in the supplied workbook; "
            "no supervised failure probability is available."
        ),
    }

    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": (
                    "I’m ScaleWise Copilot. I can explain the current scale-up "
                    "prediction, risk warnings, engineering variables and "
                    "recommended validation. What would you like to know?"
                ),
            }
        ]

    for message in st.session_state.copilot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask ScaleWise Copilot about this scale-up scenario…"
    )

    if user_question:
        st.session_state.copilot_messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("ScaleWise Copilot is analyzing the current run…"):
                answer = copilot_answer(
                    user_question,
                    context,
                    st.session_state.copilot_messages,
                )
            st.markdown(answer)

        st.session_state.copilot_messages.append(
            {"role": "assistant", "content": answer}
        )


# -----------------------------
# Download current assessment
# -----------------------------
assessment = {
    "target_scale_L": target_scale,
    "cell_line": cell_line,
    "medium": medium,
    "serum_condition": serum_condition,
    "process_stage": process_stage,
    "risk_score": risk["risk_score"],
    "risk_level": risk["risk_level"],
    "assessment_confidence_percent": risk["confidence"],
    "risk_factors": "; ".join(f["message"] for f in risk["factors"]),
    "suggested_action": risk["suggested_action"],
    "recommended_validation": "; ".join(risk["recommended_validation"]),
    **physics,
    **predictions.iloc[0].to_dict(),
}

st.sidebar.divider()
st.sidebar.download_button(
    "⬇️ Download current assessment",
    data=pd.DataFrame([assessment]).to_csv(index=False),
    file_name="scalewise_current_assessment.csv",
    mime="text/csv",
)
