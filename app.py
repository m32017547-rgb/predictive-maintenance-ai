import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, IsolationForest
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------------
# Page setup & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #0b1220; color: #e5edf8; }
section[data-testid="stSidebar"] { background: #111c2e; border-right: 1px solid #24344d; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
h1,h2,h3 { color: #f4f8ff !important; }
div[data-testid="stMetric"] { background: linear-gradient(145deg,#17263b,#111c2e); border: 1px solid #2b405d; padding: 16px; border-radius: 14px; }
div[data-testid="stMetric"] label { color: #9eb1ca !important; }
div[data-testid="stMetricValue"] { color: #f4f8ff !important; }
div.stButton > button { background: #16a6a1; color: white; border: 0; border-radius: 9px; font-weight: 700; }
div.stButton > button:hover { background: #128781; color: white; }
[data-testid="stFileUploader"] { background: #132238; border: 1px dashed #3b5878; border-radius: 12px; padding: 12px; }
.card { background:#132238; border:1px solid #293e5a; padding:18px; border-radius:14px; }
.small-muted { color:#91a6c1; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FAILURE = "Failure_Within_7_Days"
RUL = "Remaining_Useful_Life_days"
TARGETS = [FAILURE, RUL]          # BOTH are always removed from the features
ID_COLS = ["Machine_ID"]
NON_NEGATIVE = ["Temperature_C", "Vibration_mms", "Sound_dB",
                "Power_Consumption_kW", "Coolant_Flow_L_min"]
DEFAULT_FILE = "factory_sensor_simulator_2040.csv"
APP_DIR = Path(__file__).resolve().parent
PLOT_SAMPLE = 30_000              # max points drawn in scatter / histogram plots
FAIL_LABELS = {0: "No failure", 1: "Failure within 7 days"}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def show_plot(fig):
    st.plotly_chart(fig, width="stretch")


def show_table(frame, **kwargs):
    st.dataframe(frame, width="stretch", **kwargs)


def sample_df(frame, n=PLOT_SAMPLE, seed=42):
    return frame if len(frame) <= n else frame.sample(n=n, random_state=seed)


def to_binary(series):
    """Turn a True/False, 0/1 or 'yes'/'no' column into 0/1 (NaN where unknown)."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").round()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0,
               "y": 1, "n": 0, "1.0": 1, "0.0": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def numeric_features(frame):
    """Numeric sensor columns usable for unsupervised analysis (no IDs / targets)."""
    drop = [c for c in TARGETS + ID_COLS if c in frame.columns]
    return frame.select_dtypes(include=np.number).drop(columns=drop, errors="ignore")


def clean_frame(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    for c in NON_NEGATIVE:
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].clip(lower=0)
    for c in out.columns:               # True/False columns -> 0/1
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int)
    return out


def load_data(file_or_path, name):
    if name.lower().endswith(".csv"):
        return pd.read_csv(file_or_path)
    return pd.read_excel(file_or_path)


def reset_results():
    for key in ["classification_result", "regression_result", "anomaly",
                "clusters", "importance", "shap_importance", "csv_cache",
                "live_result"]:
        st.session_state.pop(key, None)


# ----------------------------------------------------------------------------
# Feature engineering & model pipelines
# ----------------------------------------------------------------------------
def add_features(df, ref_year=None):
    """Derived features. It lives INSIDE the pipeline, so raw rows can be scored."""
    x = df.copy()
    if "Installation_Year" in x.columns:
        year = pd.to_numeric(x["Installation_Year"], errors="coerce")
        ref = ref_year if ref_year is not None else year.max()
        x["Machine_Age_Years"] = (ref - year).clip(lower=0)
        x = x.drop(columns=["Installation_Year"])
    if {"Temperature_C", "Vibration_mms"}.issubset(x.columns):
        x["Temp_Vibration_Index"] = x["Temperature_C"] * x["Vibration_mms"]
    if {"Power_Consumption_kW", "Operational_Hours"}.issubset(x.columns):
        x["Power_Per_Operating_Hour"] = (
            x["Power_Consumption_kW"] / (x["Operational_Hours"].abs() + 1))
    if {"Oil_Level_pct", "Coolant_Level_pct"}.issubset(x.columns):
        x["Fluid_Level_Mean"] = x[["Oil_Level_pct", "Coolant_Level_pct"]].mean(axis=1)
    return x


def make_preprocessor(frame):
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        sparse_threshold=0,
    )


def build_pipeline(X_train, model, ref_year):
    fe = FunctionTransformer(add_features, kw_args={"ref_year": ref_year},
                             validate=False)
    prep = make_preprocessor(add_features(X_train.head(2000), ref_year))
    return Pipeline([("fe", fe), ("prep", prep), ("model", model)])


def classifier_options(pos_weight=1.0):
    opts = {"Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1)}
    try:
        from xgboost import XGBClassifier
        opts["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", tree_method="hist",
            scale_pos_weight=pos_weight, random_state=42, n_jobs=-1)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        opts["LightGBM"] = LGBMClassifier(
            n_estimators=250, learning_rate=0.05, class_weight="balanced",
            random_state=42, verbosity=-1)
    except Exception:
        pass
    return opts


def regressor_options():
    opts = {"Random Forest": RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        opts["XGBoost"] = XGBRegressor(
            n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", random_state=42, n_jobs=-1)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        opts["LightGBM"] = LGBMRegressor(
            n_estimators=250, learning_rate=0.05, random_state=42, verbosity=-1)
    except Exception:
        pass
    return opts


def build_schema(X):
    """Describe the raw input columns so the Live Prediction form can be built."""
    schema = {}
    for c in X.columns:
        s = X[c]
        info = {"missing": float(s.isna().mean())}
        vals = s.dropna()
        if pd.api.types.is_numeric_dtype(s):
            uniq = vals.unique()
            if len(uniq) <= 2:
                info.update(kind="choice", options=sorted(float(u) for u in uniq))
            else:
                info.update(
                    kind="int" if pd.api.types.is_integer_dtype(s) else "float",
                    min=float(vals.min()), max=float(vals.max()),
                    default=float(vals.median()))
        else:
            info.update(kind="choice",
                        options=sorted(vals.astype(str).unique().tolist()))
        schema[c] = info
    return schema


def prepare_xy(data, target):
    """Features = everything except BOTH targets and identifiers (no leakage)."""
    drop = [c for c in TARGETS + ID_COLS if c in data.columns]
    X = data.drop(columns=drop)
    if target == FAILURE:
        y = to_binary(data[target])
    else:
        y = pd.to_numeric(data[target], errors="coerce")
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def train_classifier(data, model_name, n_rows):
    X, y = prepare_xy(data, FAILURE)
    y = y.astype(int)
    if y.nunique() < 2:
        raise ValueError("عمود الهدف يحتوي على فئة واحدة فقط؛ لا يمكن تدريب نموذج تصنيف.")
    if n_rows < len(X):
        idx = X.sample(n=n_rows, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]
    stratify = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify)
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    candidates = classifier_options(pos_weight)
    if model_name not in candidates:
        model_name = "Random Forest"
    ref_year = (pd.to_numeric(Xtr["Installation_Year"], errors="coerce").max()
                if "Installation_Year" in Xtr.columns else None)
    pipe = build_pipeline(Xtr, candidates[model_name], ref_year)
    pipe.fit(Xtr, ytr)
    return {
        "pipeline": pipe, "X_test": Xte, "y_test": yte,
        "proba": pipe.predict_proba(Xte)[:, 1], "model": model_name,
        "n_train": len(Xtr), "schema": build_schema(Xtr),
    }


def train_regressor(data, model_name, n_rows):
    X, y = prepare_xy(data, RUL)
    if n_rows < len(X):
        idx = X.sample(n=n_rows, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = regressor_options()
    if model_name not in candidates:
        model_name = "Random Forest"
    ref_year = (pd.to_numeric(Xtr["Installation_Year"], errors="coerce").max()
                if "Installation_Year" in Xtr.columns else None)
    pipe = build_pipeline(Xtr, candidates[model_name], ref_year)
    pipe.fit(Xtr, ytr)
    pred = np.clip(pipe.predict(Xte), 0, None)
    return {
        "pipeline": pipe, "X_test": Xte, "y_test": yte, "pred": pred,
        "mae": mean_absolute_error(yte, pred),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": r2_score(yte, pred), "model": model_name,
        "n_train": len(Xtr), "schema": build_schema(Xtr),
    }


def clf_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "pred": pred,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def sample_size_widget(label, n_rows, key, default=100_000):
    if n_rows <= 5_000:
        return n_rows
    return st.slider(label, 5_000, n_rows, min(default, n_rows),
                     step=5_000 if n_rows > 20_000 else 1_000, key=key,
                     help="كل ما العدد يزيد الدقة بتتحسن شوية لكن التدريب بياخد وقت أكتر.")


def risk_gauge(prob):
    color = "#2ecc71" if prob < 0.3 else ("#f1c40f" if prob < 0.6 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Failure risk score (next 7 days)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 30], "color": "#16324a"},
                         {"range": [30, 60], "color": "#3a3a1f"},
                         {"range": [60, 100], "color": "#4a1f24"}]}))
    fig.update_layout(template="plotly_dark", height=300,
                      margin=dict(l=20, r=20, t=60, b=10))
    return fig


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("data", None)
st.session_state.setdefault("data_key", None)
st.session_state.setdefault("classification_result", None)
st.session_state.setdefault("regression_result", None)


def set_dataset(raw, key):
    st.session_state.data = clean_frame(raw)
    st.session_state.data_key = key
    reset_results()
    st.session_state.setdefault("classification_result", None)
    st.session_state.setdefault("regression_result", None)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ PM AI Platform")
    st.caption("Industrial intelligence • Predict before failure")
    page = st.radio("NAVIGATION", [
        "Overview", "Data Explorer", "Failure Prediction",
        "Useful Life Regression", "Live Prediction", "Anomaly Detection",
        "Machine Clustering", "PCA Analysis", "Model Explainability",
        "Export & Reports"
    ])
    st.divider()
    st.markdown("### Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        key = (uploaded.name, uploaded.size)
        # Load only when the file really changed (not on every rerun).
        if st.session_state.data_key != key:
            try:
                with st.spinner("Reading file..."):
                    set_dataset(load_data(uploaded, uploaded.name), key)
            except Exception as e:
                st.error(f"Could not read file: {e}")
    local_file = APP_DIR / DEFAULT_FILE
    if local_file.exists():
        if st.button(f"Load {DEFAULT_FILE}"):
            with st.spinner("Reading file..."):
                set_dataset(load_data(local_file, local_file.name), ("local", DEFAULT_FILE))
    if st.session_state.data is not None:
        d = st.session_state.data
        st.caption(f"{len(d):,} rows × {len(d.columns)} columns")

st.markdown("# Predictive Maintenance")
st.markdown("<div class='small-muted'>AI-powered equipment health monitoring, "
            "failure prediction and maintenance intelligence</div>",
            unsafe_allow_html=True)

df = st.session_state.data

if df is None:
    st.info("ابدأ برفع ملف بيانات المصنع CSV أو Excel من القائمة الجانبية. "
            f"ولو حطيت الملف `{DEFAULT_FILE}` جنب `app.py` هيظهر زرار تحميل سريع.")
    st.markdown("### What this platform can do")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failure Prediction", "Classification")
    c2.metric("Useful Life", "Regression")
    c3.metric("Anomalies", "Isolation Forest")
    c4.metric("Segments", "K-Means")
    st.markdown("""
    <div class="card">
    <h3>Expected dataset fields</h3>
    <p>For the full project workflow, the dataset should include <b>Failure_Within_7_Days</b> and/or <b>Remaining_Useful_Life_days</b>, along with sensor measurements such as temperature, vibration, sound, oil, coolant, power consumption and operating hours.</p>
    <p class="small-muted">You can still use Data Explorer and unsupervised analysis with a dataset that has no target column.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

has_fail = FAILURE in df.columns
has_rul = RUL in df.columns

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "Overview":
    st.markdown("## Operations overview")
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    a, b, c, d = st.columns(4)
    a.metric("Records", f"{rows:,}")
    b.metric("Features", cols)
    c.metric("Missing cells", f"{missing:,}")
    if has_fail:
        rate = to_binary(df[FAILURE]).mean()
        d.metric("Failure rate (7 days)", f"{rate:.1%}")
    else:
        d.metric("Failure target", "Not found")
    if has_rul:
        e1, e2, e3 = st.columns(3)
        e1.metric("Mean remaining life", f"{df[RUL].mean():,.0f} days")
        e2.metric("Median remaining life", f"{df[RUL].median():,.0f} days")
        e3.metric("Machines with RUL ≤ 30 days", f"{int((df[RUL] <= 30).sum()):,}")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Dataset preview")
        show_table(df.head(12))
    with right:
        st.markdown("### Missing values (%)")
        miss = (df.isna().mean() * 100).sort_values(ascending=False).head(12)
        miss = miss[miss > 0]
        if len(miss):
            show_plot(px.bar(x=miss.values, y=miss.index, orientation="h",
                             labels={"x": "Missing %", "y": "Feature"},
                             template="plotly_dark"))
        else:
            st.success("No missing values detected.")
    if has_fail:
        l2, r2 = st.columns(2)
        with l2:
            st.markdown("### Failure target distribution")
            vc = (to_binary(df[FAILURE]).map(FAIL_LABELS).value_counts()
                  .rename_axis("Class").reset_index(name="Count"))
            show_plot(px.pie(vc, names="Class", values="Count", hole=.55,
                             template="plotly_dark"))
        with r2:
            if "Machine_Type" in df.columns:
                st.markdown("### Failure rate by machine type")
                rate = (df.assign(_f=to_binary(df[FAILURE]))
                        .groupby("Machine_Type")["_f"].mean().sort_values() * 100)
                show_plot(px.bar(x=rate.values, y=rate.index, orientation="h",
                                 labels={"x": "Failure rate %", "y": "Machine type"},
                                 template="plotly_dark"))
    else:
        st.warning(f"Target column '{FAILURE}' is not present. Upload the matching "
                   "labeled dataset for supervised prediction.")

elif page == "Data Explorer":
    st.markdown("## Data Explorer")
    st.caption(f"Showing the first 1,000 of {len(df):,} rows (the full file is used for the statistics below).")
    show_table(df.head(1000), height=380)
    st.markdown("### Summary statistics")
    show_table(df.describe().T)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Data types")
        types = df.dtypes.astype(str).value_counts().reset_index()
        types.columns = ["Data type", "Columns"]
        show_plot(px.bar(types, x="Data type", y="Columns", template="plotly_dark"))
    with c2:
        st.markdown("### Numeric distributions")
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            chosen = st.selectbox("Choose a numeric feature", numeric)
            show_plot(px.histogram(sample_df(df), x=chosen, marginal="box",
                                   template="plotly_dark"))
    st.markdown("### Correlation matrix")
    num = df.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        show_plot(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            aspect="auto", template="plotly_dark"))

elif page == "Failure Prediction":
    st.markdown("## Failure prediction")
    if not has_fail:
        st.error(f"Required target column '{FAILURE}' was not found in this dataset.")
        st.stop()
    st.caption(f"`{RUL}` and `Machine_ID` are excluded from the features on purpose: "
               "the remaining life directly reveals the failure label (data leakage).")
    c1, c2 = st.columns([1, 2])
    with c1:
        selected = st.selectbox("Classification model", list(classifier_options().keys()))
    with c2:
        n_rows = sample_size_widget("Rows used for training + testing", len(df), "clf_rows")
    if st.button("Train failure prediction model", type="primary"):
        with st.spinner("Training and evaluating model..."):
            try:
                st.session_state.classification_result = train_classifier(df, selected, n_rows)
                st.session_state.pop("importance", None)
                st.session_state.pop("shap_importance", None)
            except Exception as e:
                st.error(f"Training failed: {e}")
    res = st.session_state.classification_result
    if res:
        st.success(f"{res['model']} trained on {res['n_train']:,} rows. "
                   f"Test set = {len(res['y_test']):,} held-out rows.")
        thr = st.slider("Decision threshold (lower = catches more failures, more false alarms)",
                        0.05, 0.95, 0.50, 0.05, key="thr")
        m = clf_metrics(res["y_test"], res["proba"], thr)
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{m['accuracy']:.3f}")
        b.metric("Precision (failure)", f"{m['precision']:.3f}")
        c.metric("Recall (failure)", f"{m['recall']:.3f}")
        d.metric("F1 (failure)", f"{m['f1']:.3f}")
        e.metric("ROC AUC", f"{roc_auc_score(res['y_test'], res['proba']):.3f}")
        st.caption("الأرقام الأساسية هنا خاصة بفئة الفشل (Failure) لأن الفئة دي نادرة؛ "
                   "الـ Accuracy لوحدها بتضلل مع بيانات غير متوازنة.")
        left, right = st.columns(2)
        with left:
            cm = confusion_matrix(res["y_test"], m["pred"], labels=[0, 1])
            show_plot(px.imshow(cm, x=["No failure", "Failure"], y=["No failure", "Failure"],
                                text_auto=True, color_continuous_scale="Teal",
                                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                                title="Confusion matrix", template="plotly_dark"))
        with right:
            fpr, tpr, _ = roc_curve(res["y_test"], res["proba"])
            prec, rec, _ = precision_recall_curve(res["y_test"], res["proba"])
            curve = st.radio("Curve", ["ROC", "Precision-Recall"], horizontal=True)
            if curve == "ROC":
                fig = px.area(x=fpr, y=tpr, labels={"x": "False positive rate", "y": "True positive rate"},
                              title="ROC curve", template="plotly_dark")
            else:
                ap = average_precision_score(res["y_test"], res["proba"])
                fig = px.area(x=rec, y=prec, labels={"x": "Recall", "y": "Precision"},
                              title=f"Precision-Recall curve (AP = {ap:.3f})",
                              template="plotly_dark")
            show_plot(fig)
        st.markdown("### Classification report")
        report = classification_report(res["y_test"], m["pred"], output_dict=True,
                                       zero_division=0)
        show_table(pd.DataFrame(report).T)
        out = res["X_test"].copy()
        out["Actual"] = res["y_test"].values
        out["Risk_Score"] = res["proba"]
        out["Predicted"] = m["pred"]
        st.download_button("Download test predictions CSV",
                           out.to_csv(index=False).encode("utf-8-sig"),
                           "failure_predictions.csv", "text/csv")

elif page == "Useful Life Regression":
    st.markdown("## Remaining useful life (RUL) regression")
    if not has_rul:
        st.error(f"Required regression target '{RUL}' was not found in this dataset.")
        st.stop()
    st.caption(f"`{FAILURE}` and `Machine_ID` are excluded from the features on purpose "
               "(the failure flag is derived from the remaining life).")
    c1, c2 = st.columns([1, 2])
    with c1:
        selected = st.selectbox("Regression model", list(regressor_options().keys()))
    with c2:
        n_rows = sample_size_widget("Rows used for training + testing", len(df), "reg_rows", default=50_000)
    if st.button("Train RUL regression model", type="primary"):
        with st.spinner("Training regression model..."):
            try:
                st.session_state.regression_result = train_regressor(df, selected, n_rows)
                st.session_state.pop("importance", None)
                st.session_state.pop("shap_importance", None)
            except Exception as e:
                st.error(f"Training failed: {e}")
    res = st.session_state.regression_result
    if res:
        st.success(f"{res['model']} trained on {res['n_train']:,} rows. "
                   f"Test set = {len(res['y_test']):,} held-out rows.")
        a, b, c = st.columns(3)
        a.metric("MAE", f"{res['mae']:.1f} days")
        b.metric("RMSE", f"{res['rmse']:.1f} days")
        c.metric("R²", f"{res['r2']:.3f}")
        plot = pd.DataFrame({"Actual": res["y_test"].values, "Predicted": res["pred"]})
        l, r = st.columns(2)
        with l:
            show_plot(px.scatter(sample_df(plot, 5000), x="Actual", y="Predicted",
                                 opacity=.5, title="Actual vs Predicted RUL (5,000 sample)",
                                 template="plotly_dark"))
        with r:
            plot["Error"] = plot["Predicted"] - plot["Actual"]
            show_plot(px.histogram(plot, x="Error", nbins=60, title="Prediction error (days)",
                                   template="plotly_dark"))
        st.download_button("Download RUL predictions CSV",
                           plot.to_csv(index=False).encode("utf-8-sig"),
                           "rul_predictions.csv", "text/csv")

elif page == "Live Prediction":
    st.markdown("## Live prediction for a single machine")
    clf = st.session_state.classification_result
    reg = st.session_state.regression_result
    if not clf and not reg:
        st.info("درّب موديل الأول من صفحة Failure Prediction أو Useful Life Regression، "
                "وبعدين ارجع هنا تدخل قراءات أي ماكينة وتشوف التوقع.")
        st.stop()
    schema = (clf or reg)["schema"]
    with st.form("live_form"):
        st.markdown("Enter the machine readings:")
        grid = st.columns(3)
        values = {}
        for i, (col, meta) in enumerate(schema.items()):
            with grid[i % 3]:
                optional = meta["missing"] > 0.5
                use = True
                if optional:
                    use = st.checkbox(f"Machine has {col}", value=False, key=f"use_{col}",
                                      help="Only some machine types have this sensor.")
                if meta["kind"] == "choice":
                    opts = meta["options"]
                    v = st.selectbox(col, opts, key=f"in_{col}")
                elif meta["kind"] == "int":
                    v = st.number_input(col, min_value=int(meta["min"]), max_value=int(meta["max"]),
                                        value=int(meta["default"]), step=1, key=f"in_{col}")
                else:
                    v = st.number_input(col, min_value=meta["min"], max_value=meta["max"],
                                        value=meta["default"], key=f"in_{col}")
                values[col] = v if use else np.nan
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        row = pd.DataFrame([values])
        for col, meta in schema.items():
            if meta["kind"] != "choice" or (meta["options"] and not isinstance(meta["options"][0], str)):
                row[col] = pd.to_numeric(row[col], errors="coerce")
        result = {}
        if clf:
            result["risk"] = float(clf["pipeline"].predict_proba(row)[0, 1])
        if reg:
            result["rul"] = float(max(reg["pipeline"].predict(row)[0], 0))
        st.session_state.live_result = result
    result = st.session_state.get("live_result")
    if result:
        l, r = st.columns([1.2, 1])
        with l:
            if "risk" in result:
                show_plot(risk_gauge(result["risk"]))
                st.caption("Risk score مش احتمال حقيقي مضبوط؛ الموديل متدرب بأوزان للفئة النادرة "
                           "فبيميل يرفع الدرجة. استخدمه للترتيب والمقارنة بين الماكينات.")
        with r:
            if "rul" in result:
                st.metric("Estimated remaining useful life", f"{result['rul']:,.0f} days")
            if "risk" in result:
                if result["risk"] >= 0.6:
                    st.error("High risk — schedule an inspection within the next few days.")
                elif result["risk"] >= 0.3:
                    st.warning("Medium risk — plan a maintenance check soon.")
                else:
                    st.success("Low risk — continue normal monitoring.")
    if clf:
        st.markdown("### Top 25 highest-risk machines (held-out test set)")
        top = pd.DataFrame({"Risk_Score": clf["proba"], "Actual_Failure": clf["y_test"].values},
                           index=clf["X_test"].index).sort_values("Risk_Score", ascending=False).head(25)
        info_cols = [c for c in ID_COLS + ["Machine_Type", "Temperature_C", "Vibration_mms",
                                           "Error_Codes_Last_30_Days"] if c in df.columns]
        show_table(df.loc[top.index, info_cols].join(top).reset_index(drop=True))

elif page == "Anomaly Detection":
    st.markdown("## Anomaly detection")
    numeric = numeric_features(df)
    if numeric.shape[1] < 1:
        st.warning("No numeric sensor columns are available.")
    else:
        contamination = st.slider("Expected anomaly fraction", 0.01, 0.20, 0.05, 0.01)
        if st.button("Run Isolation Forest", type="primary"):
            with st.spinner("Scoring all records..."):
                arr = SimpleImputer(strategy="median").fit_transform(numeric)
                iso = IsolationForest(contamination=contamination, random_state=42,
                                      n_estimators=200, n_jobs=-1)
                flag = iso.fit_predict(arr) == -1
                score = iso.decision_function(arr)
                scored = df.assign(Anomaly_Score=score, Is_Anomaly=flag)
                anomalies = scored[scored["Is_Anomaly"]].sort_values("Anomaly_Score")
                st.session_state.anomaly = {
                    "scores": pd.Series(score, index=df.index),
                    "flag": pd.Series(flag, index=df.index),
                    "top": anomalies.head(100),
                    "csv": anomalies.to_csv(index=False).encode("utf-8-sig"),
                }
        res = st.session_state.get("anomaly")
        if res:
            n_anom = int(res["flag"].sum())
            a, b, c = st.columns(3)
            a.metric("Anomalies detected", f"{n_anom:,}")
            b.metric("Normal records", f"{len(df) - n_anom:,}")
            if has_fail:
                f = to_binary(df[FAILURE])
                ratio = f[res["flag"]].mean() / max(f[~res["flag"]].mean(), 1e-9)
                c.metric("Failure rate: anomalies vs normal", f"{ratio:.1f}x")
            plot = pd.DataFrame({"Anomaly_Score": res["scores"], "Status":
                                 np.where(res["flag"], "Anomaly", "Normal")})
            show_plot(px.histogram(sample_df(plot, 50_000), x="Anomaly_Score", color="Status",
                                   template="plotly_dark", title="Anomaly score distribution"))
            st.markdown("### 100 most anomalous machines")
            show_table(res["top"])
            st.download_button("Download all anomalies", res["csv"], "anomaly_results.csv", "text/csv")

elif page == "Machine Clustering":
    st.markdown("## Machine segmentation — K-Means")
    numeric = numeric_features(df)
    if numeric.shape[1] < 2:
        st.warning("At least two numeric features are required.")
    else:
        k = st.slider("Number of clusters (K)", 2, 10, 3)
        if st.button("Run K-Means", type="primary"):
            with st.spinner("Clustering..."):
                sub = sample_df(numeric, 100_000)
                imp = SimpleImputer(strategy="median").fit(sub)
                sc = StandardScaler().fit(imp.transform(sub))
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                km.fit(sc.transform(imp.transform(sub)))
                full = sc.transform(imp.transform(numeric))
                labels = km.predict(full)
                sil_idx = np.random.RandomState(42).choice(len(full), min(10_000, len(full)),
                                                          replace=False)
                st.session_state.clusters = {
                    "labels": pd.Series(labels.astype(str), index=df.index),
                    "silhouette": silhouette_score(full[sil_idx], labels[sil_idx]),
                }
        res = st.session_state.get("clusters")
        if res:
            labels = res["labels"]
            a, b = st.columns(2)
            a.metric("Silhouette score", f"{res['silhouette']:.3f}")
            b.metric("Clusters", labels.nunique())
            plot = sample_df(numeric.assign(Machine_Cluster=labels))
            c1, c2 = st.columns(2)
            cols = numeric.columns.tolist()
            xcol = c1.selectbox("X axis", cols, index=0)
            ycol = c2.selectbox("Y axis", cols, index=min(1, len(cols) - 1))
            show_plot(px.scatter(plot, x=xcol, y=ycol, color="Machine_Cluster",
                                 opacity=.6, template="plotly_dark"))
            st.markdown("### Cluster profiles (mean values)")
            prof_cols = cols + [c for c in TARGETS if c in df.columns]
            profile = df[prof_cols].assign(Machine_Cluster=labels).groupby("Machine_Cluster").mean()
            profile.insert(0, "Machines", labels.value_counts().sort_index())
            show_table(profile.round(2))
            st.download_button(
                "Download cluster labels",
                labels.rename("Machine_Cluster").reset_index().to_csv(index=False).encode("utf-8-sig"),
                "machine_clusters.csv", "text/csv")

elif page == "PCA Analysis":
    st.markdown("## Principal Component Analysis")
    numeric = numeric_features(df)
    if numeric.shape[1] < 2:
        st.warning("PCA requires at least two numeric features.")
    else:
        sub = sample_df(numeric, 100_000)
        arr = SimpleImputer(strategy="median").fit_transform(sub)
        arr = StandardScaler().fit_transform(arr)
        n_comp = min(6, arr.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        coords = pca.fit_transform(arr)
        pc = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"], index=sub.index)
        st.metric("Variance explained by first 2 PCs", f"{pca.explained_variance_ratio_[:2].sum():.1%}")
        color = None
        if has_fail:
            pc["Failure"] = to_binary(df.loc[sub.index, FAILURE]).map(FAIL_LABELS).values
            color = "Failure"
        show_plot(px.scatter(sample_df(pc, 20_000), x="PC1", y="PC2", color=color, opacity=.6,
                             template="plotly_dark", title="PCA projection"))
        variance = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(n_comp)],
                                 "Explained variance": pca.explained_variance_ratio_})
        show_plot(px.bar(variance, x="Component", y="Explained variance", template="plotly_dark"))

elif page == "Model Explainability":
    st.markdown("## Model explainability")
    available = {}
    if st.session_state.classification_result:
        available["Failure classifier"] = ("clf", st.session_state.classification_result)
    if st.session_state.regression_result:
        available["RUL regressor"] = ("reg", st.session_state.regression_result)
    if not available:
        st.info("Train a failure prediction or RUL regression model first.")
        st.stop()
    choice = st.radio("Model", list(available.keys()), horizontal=True)
    task, res = available[choice]
    pipe, Xtest = res["pipeline"], res["X_test"]
    st.write(f"Current model: **{res['model']}**")

    st.markdown("### Permutation importance")
    if st.button("Calculate permutation importance", type="primary"):
        with st.spinner("Calculating feature importance..."):
            try:
                sub = sample_df(Xtest, 5000)
                ysub = res["y_test"].loc[sub.index]
                scoring = "average_precision" if task == "clf" else "r2"
                imp = permutation_importance(pipe, sub, ysub, n_repeats=3, random_state=42,
                                             scoring=scoring, n_jobs=-1)
                st.session_state.importance = (task, pd.DataFrame({
                    "Feature": sub.columns, "Importance": imp.importances_mean
                }).sort_values("Importance", ascending=False).head(20))
            except Exception as e:
                st.error(f"Importance calculation failed: {e}")
    imp_state = st.session_state.get("importance")
    if imp_state and imp_state[0] == task:
        show_plot(px.bar(imp_state[1].sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", template="plotly_dark",
                         title="Permutation feature importance"))

    st.markdown("### SHAP")
    if st.button("Run SHAP explanation"):
        with st.spinner("Computing SHAP values on 100 test rows..."):
            try:
                import shap
                sub = sample_df(Xtest, 100)
                transformed = pipe[:-1].transform(sub)
                names = pipe.named_steps["prep"].get_feature_names_out()
                explainer = shap.TreeExplainer(pipe.named_steps["model"])
                sv = explainer.shap_values(transformed, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[-1]
                sv = np.asarray(sv)
                if sv.ndim == 3:
                    sv = sv[:, :, -1]
                st.session_state.shap_importance = (task, pd.DataFrame({
                    "Feature": names, "Mean |SHAP|": np.abs(sv).mean(axis=0)
                }).sort_values("Mean |SHAP|", ascending=False).head(20))
            except ImportError:
                st.warning("SHAP is not installed. Run: pip install shap")
            except Exception as e:
                st.error(f"SHAP explanation could not be generated for this model: {e}")
    shap_state = st.session_state.get("shap_importance")
    if shap_state and shap_state[0] == task:
        show_plot(px.bar(shap_state[1].sort_values("Mean |SHAP|"), x="Mean |SHAP|", y="Feature",
                         orientation="h", template="plotly_dark",
                         title="Mean absolute SHAP value"))
    st.caption("Permutation importance and SHAP are two different views of which features "
               "drive the model's predictions.")

elif page == "Export & Reports":
    st.markdown("## Export & reports")
    st.write("Download the cleaned dataset or model outputs generated during this session.")
    if st.button("Prepare cleaned dataset CSV"):
        with st.spinner("Building file..."):
            st.session_state.csv_cache = df.to_csv(index=False).encode("utf-8-sig")
    if st.session_state.get("csv_cache"):
        st.download_button("Download cleaned dataset CSV", st.session_state.csv_cache,
                           "cleaned_machine_data.csv", "text/csv")
    if st.session_state.classification_result:
        r = st.session_state.classification_result
        m = clf_metrics(r["y_test"], r["proba"], 0.5)
        summary = pd.DataFrame([{
            "Task": "Failure classification", "Model": r["model"], "Threshold": 0.5,
            "Accuracy": m["accuracy"], "Precision_failure": m["precision"],
            "Recall_failure": m["recall"], "F1_failure": m["f1"],
            "ROC_AUC": roc_auc_score(r["y_test"], r["proba"])}])
        st.download_button("Download classification model summary",
                           summary.to_csv(index=False).encode("utf-8-sig"),
                           "classification_summary.csv", "text/csv")
    if st.session_state.regression_result:
        r = st.session_state.regression_result
        summary = pd.DataFrame([{
            "Task": "RUL regression", "Model": r["model"],
            "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"]}])
        st.download_button("Download regression model summary",
                           summary.to_csv(index=False).encode("utf-8-sig"),
                           "regression_summary.csv", "text/csv")

st.markdown("---")
st.markdown("<div class='small-muted'>Predictive Maintenance AI Platform • Educational and "
            "decision-support prototype. Validate models on representative industrial data "
            "before operational use.</div>", unsafe_allow_html=True)
