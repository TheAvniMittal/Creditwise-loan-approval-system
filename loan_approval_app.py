import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, precision_score, accuracy_score,
    recall_score, f1_score, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* gradient header */
.hero {
    background: linear-gradient(135deg, #1a237e 0%, #283593 40%, #1565c0 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(21,101,192,0.25);
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1.05rem; opacity: 0.85; margin: 0.5rem 0 0; }

/* metric cards */
.metric-card {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(13,71,161,0.3);
}
.metric-card .value { font-size: 2rem; font-weight: 700; }
.metric-card .label { font-size: 0.8rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* result boxes */
.approved-box {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: white; border-radius: 16px; padding: 2rem;
    text-align: center; font-size: 1.3rem; font-weight: 600;
    box-shadow: 0 6px 24px rgba(46,125,50,0.35);
}
.rejected-box {
    background: linear-gradient(135deg, #b71c1c, #c62828);
    color: white; border-radius: 16px; padding: 2rem;
    text-align: center; font-size: 1.3rem; font-weight: 600;
    box-shadow: 0 6px 24px rgba(198,40,40,0.35);
}

/* section title */
.section-title {
    font-size: 1.25rem; font-weight: 700; color: #1565c0;
    border-left: 4px solid #1565c0; padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* sidebar polish */
[data-testid="stSidebar"] { background: #f0f4ff; }

/* stButton primary */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2rem; font-weight: 600; font-size: 1rem;
    box-shadow: 0 4px 14px rgba(13,71,161,0.35);
    transition: transform .15s, box-shadow .15s;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(13,71,161,0.45);
}

/* tab underline */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; font-weight: 600;
    padding: 0.5rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols   = df.select_dtypes(include=["number"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])

    return df


@st.cache_data
def train_models(df):
    raw = df.copy()

    # Feature engineering
    raw["DTI_Ratio_sq"]    = raw["DTI_Ratio"] ** 2
    raw["Credit_Score_sq"] = raw["Credit_Score"] ** 2

    drop_cols = ["Applicant_ID"] if "Applicant_ID" in raw.columns else []
    raw = raw.drop(columns=drop_cols, errors="ignore")

    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    ohe_cols = [c for c in ohe_cols if c in raw.columns]

    ohe     = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(raw[ohe_cols])
    enc_df  = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=raw.index)
    raw     = pd.concat([raw.drop(columns=ohe_cols), enc_df], axis=1)

    X = raw.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"], errors="ignore")
    y = raw["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_test_sc    = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=9)":           KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes":         GaussianNB(),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_sc)[:, 1])
        results[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1 Score":  f1_score(y_test, y_pred),
            "AUC":       auc(fpr, tpr),
            "cm":        confusion_matrix(y_test, y_pred),
            "fpr":       fpr,
            "tpr":       tpr,
        }
        trained[name] = model

    return results, trained, scaler, X.columns.tolist(), ohe, ohe_cols


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏦 Loan Approval Intelligence</h1>
  <p>Upload your dataset · Explore patterns · Train models · Predict instantly</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.markdown("## 📂 Data Source")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.markdown("---")
    st.markdown("**Quick Guide**")
    st.info("1️⃣ Upload your loan CSV\n\n2️⃣ Explore EDA charts\n\n3️⃣ Compare ML models\n\n4️⃣ Predict any applicant")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + scikit-learn")

# ── Require upload ────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding:3rem; background:#f0f4ff;
                border-radius:16px; border:2px dashed #90caf9; margin-top:2rem;'>
      <div style='font-size:4rem;'>📊</div>
      <h3 style='color:#1565c0;'>No dataset loaded yet</h3>
      <p style='color:#555;'>Upload your <b>loan_approval_data.csv</b> from the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = load_and_preprocess(uploaded)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 EDA", "🤖 Model Comparison", "🎯 Predict"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Dataset Snapshot</div>', unsafe_allow_html=True)

    total   = len(df_raw)
    approved = int(df_raw["Loan_Approved"].sum()) if "Loan_Approved" in df_raw.columns else 0
    rejected = total - approved
    rate     = approved / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🗂️ Total Records", f"{total:,}"),
        ("✅ Approved", f"{approved:,}"),
        ("❌ Rejected", f"{rejected:,}"),
        ("📈 Approval Rate", f"{rate:.1f}%"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], cards):
        col.markdown(f"""
        <div class="metric-card">
          <div class="value">{val}</div>
          <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Raw Data Preview</div>', unsafe_allow_html=True)
    n = st.slider("Rows to display", 5, 100, 10)
    st.dataframe(df_raw.head(n), use_container_width=True)

    st.markdown('<div class="section-title">Basic Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # ── Approval donut ──────────────────────────────────────────────────────
    if "Loan_Approved" in df_raw.columns:
        counts = df_raw["Loan_Approved"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({0: "Rejected", 1: "Approved"})

        fig_donut = px.pie(
            counts, values="Count", names="Status",
            hole=0.55,
            color="Status",
            color_discrete_map={"Approved": "#2e7d32", "Rejected": "#c62828"},
            title="Loan Approval Distribution",
        )
        fig_donut.update_traces(textposition="outside", textfont_size=14)
        fig_donut.update_layout(
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=50, b=50),
        )

    # ── Numeric distributions ───────────────────────────────────────────────
    num_cols = df_raw.select_dtypes(include="number").columns.tolist()
    chosen_num = st.selectbox("Numeric column to explore", num_cols, index=num_cols.index("Loan_Amount") if "Loan_Amount" in num_cols else 0)

    fig_hist = px.histogram(
        df_raw, x=chosen_num,
        color="Loan_Approved" if "Loan_Approved" in df_raw.columns else None,
        nbins=30, barmode="overlay",
        color_discrete_map={0: "#ef9a9a", 1: "#a5d6a7"},
        title=f"Distribution of {chosen_num} by Approval Status",
        labels={"Loan_Approved": "Approved"},
        template="plotly_white",
    )
    fig_hist.update_layout(legend_title_text="Approved (1=Yes)")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(fig_donut, use_container_width=True)
    with col_b:
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Box plots ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Income / Score / Ratio Box Plots</div>', unsafe_allow_html=True)
    box_features = [c for c in ["Applicant_Income", "Credit_Score", "DTI_Ratio", "Savings", "Loan_Amount"] if c in df_raw.columns]
    chosen_box = st.multiselect("Select features", box_features, default=box_features[:4])

    if chosen_box and "Loan_Approved" in df_raw.columns:
        fig_box = make_subplots(
            rows=1, cols=len(chosen_box),
            subplot_titles=chosen_box,
        )
        colours = ["#1565c0", "#2e7d32", "#6a1b9a", "#e65100"]
        for i, feat in enumerate(chosen_box):
            for status, name, colour in [(0, "Rejected", "#ef9a9a"), (1, "Approved", "#a5d6a7")]:
                sub = df_raw[df_raw["Loan_Approved"] == status][feat]
                fig_box.add_trace(
                    go.Box(y=sub, name=name, marker_color=colour, showlegend=(i == 0)),
                    row=1, col=i + 1
                )
        fig_box.update_layout(height=420, template="plotly_white", boxmode="group",
                               legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation heatmap ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df_raw.select_dtypes(include="number").corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
        zmin=-1, zmax=1,
    )
    fig_heat.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Categorical bar ─────────────────────────────────────────────────────
    cat_cols_raw = [c for c in df_raw.select_dtypes(include="object").columns if c in ["Gender", "Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Employer_Category"]]
    if cat_cols_raw:
        st.markdown('<div class="section-title">Categorical Breakdown</div>', unsafe_allow_html=True)
        cat_choice = st.selectbox("Choose a categorical feature", cat_cols_raw)
        cat_cnt = df_raw[cat_choice].value_counts().reset_index()
        cat_cnt.columns = [cat_choice, "Count"]
        fig_bar = px.bar(
            cat_cnt, x=cat_choice, y="Count",
            color="Count", color_continuous_scale="Blues",
            title=f"{cat_choice} Distribution", template="plotly_white",
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.info("Models are trained with feature engineering (DTI² + Credit Score²) as per the original notebook.")

    with st.spinner("⚙️ Training Logistic Regression, KNN, and Naive Bayes …"):
        results, trained_models, scaler, feature_cols, ohe_obj, ohe_cols_list = train_models(df_raw)

    # ── Metrics table ───────────────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        name: {k: f"{v:.4f}" for k, v in vals.items() if k not in ("cm", "fpr", "tpr")}
        for name, vals in results.items()
    }).T.reset_index().rename(columns={"index": "Model"})

    st.dataframe(
        metrics_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
                                       color="#c8e6c9"),
        use_container_width=True,
    )

    # ── Radar chart ─────────────────────────────────────────────────────────
    metrics_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    fig_radar = go.Figure()
    colours = ["#1565c0", "#2e7d32", "#e65100"]
    for (name, vals), colour in zip(results.items(), colours):
        vals_list = [vals[k] for k in metrics_keys]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_list + [vals_list[0]],
            theta=metrics_keys + [metrics_keys[0]],
            fill="toself", name=name,
            line_color=colour, fillcolor=colour,
            opacity=0.25,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 1])),
        title="Model Performance Radar", template="plotly_white",
        legend=dict(orientation="h", y=-0.1),
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Bar comparison ──────────────────────────────────────────────────────
    sel_metric = st.selectbox("Compare models by metric", metrics_keys)
    bar_vals   = {name: results[name][sel_metric] for name in results}
    fig_bar2   = px.bar(
        x=list(bar_vals.keys()), y=list(bar_vals.values()),
        color=list(bar_vals.keys()),
        color_discrete_sequence=["#1565c0", "#2e7d32", "#e65100"],
        title=f"Model {sel_metric} Comparison",
        labels={"x": "Model", "y": sel_metric},
        template="plotly_white", text=[f"{v:.4f}" for v in bar_vals.values()],
    )
    fig_bar2.update_traces(textposition="outside")
    fig_bar2.update_layout(showlegend=False, height=350, yaxis_range=[0, 1.05])
    st.plotly_chart(fig_bar2, use_container_width=True)

    # ── ROC curves ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">ROC Curves</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(dash="dash", color="grey"))
    for (name, vals), colour in zip(results.items(), colours):
        fig_roc.add_trace(go.Scatter(
            x=vals["fpr"], y=vals["tpr"],
            mode="lines", name=f"{name} (AUC={vals['AUC']:.3f})",
            line=dict(color=colour, width=2.5),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        title="ROC Curves – All Models", template="plotly_white",
        legend=dict(x=0.55, y=0.05), height=420,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Confusion matrices ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for col, (name, vals) in zip(cm_cols, results.items()):
        cm = vals["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            color_continuous_scale="Blues",
            x=["Pred Rejected", "Pred Approved"],
            y=["True Rejected", "True Approved"],
            title=name,
        )
        fig_cm.update_layout(height=320, margin=dict(t=50, b=10, l=10, r=10),
                              coloraxis_showscale=False)
        col.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Real-Time Loan Prediction</div>', unsafe_allow_html=True)

    # Ensure models are trained
    if "trained_models" not in dir():
        with st.spinner("Preparing models …"):
            results, trained_models, scaler, feature_cols, ohe_obj, ohe_cols_list = train_models(df_raw)

    chosen_model = st.selectbox("🤖 Select Model", list(trained_models.keys()))

    st.markdown("#### 👤 Applicant Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age             = st.number_input("Age", 18, 75, 30)
        applicant_income= st.number_input("Applicant Income (₹)", 10000, 5000000, 50000, step=5000)
        loan_amount     = st.number_input("Loan Amount (₹)", 50000, 10000000, 500000, step=10000)
        loan_term       = st.number_input("Loan Term (months)", 6, 360, 60)
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

    with col2:
        credit_score    = st.slider("Credit Score", 300, 850, 700)
        dti_ratio       = st.slider("DTI Ratio (%)", 0.0, 60.0, 25.0, 0.5)
        savings         = st.number_input("Savings (₹)", 0, 10000000, 100000, step=10000)
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
        employer_category = st.selectbox("Employer Category", ["Government", "Private", "NGO"])

    with col3:
        gender          = st.selectbox("Gender", ["Male", "Female"])
        marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        property_area   = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
        loan_purpose    = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business", "Auto"])
        num_dependents  = st.number_input("Number of Dependents", 0, 10, 1)

    edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}

    if st.button("🔍 Predict Loan Approval", use_container_width=True):
        # Build input df matching training schema
        input_dict = {
            "Age":               age,
            "Applicant_Income":  applicant_income,
            "Loan_Amount":       loan_amount,
            "Loan_Term":         loan_term,
            "Education_Level":   edu_map[education_level],
            "Credit_Score":      credit_score,
            "DTI_Ratio":         dti_ratio,
            "Savings":           savings,
            "Num_Dependents":    num_dependents,
            "Employment_Status": employment_status,
            "Employer_Category": employer_category,
            "Gender":            gender,
            "Marital_Status":    marital_status,
            "Property_Area":     property_area,
            "Loan_Purpose":      loan_purpose,
        }
        input_df = pd.DataFrame([input_dict])

        # Feature engineering
        input_df["DTI_Ratio_sq"]    = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

        # OHE
        ohe_present = [c for c in ohe_cols_list if c in input_df.columns]
        encoded_inp = ohe_obj.transform(input_df[ohe_present])
        enc_inp_df  = pd.DataFrame(encoded_inp,
                                   columns=ohe_obj.get_feature_names_out(ohe_present))
        input_df = pd.concat([input_df.drop(columns=ohe_present), enc_inp_df], axis=1)

        # Drop Credit_Score & DTI_Ratio (as in training)
        input_df = input_df.drop(columns=["Credit_Score", "DTI_Ratio"], errors="ignore")

        # Align columns
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        # Scale & predict
        X_scaled   = scaler.transform(input_df)
        model      = trained_models[chosen_model]
        prediction = model.predict(X_scaled)[0]
        prob       = model.predict_proba(X_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
            <div class="approved-box">
              ✅ LOAN APPROVED<br>
              <span style='font-size:1rem;opacity:0.9;'>Confidence: {prob[1]*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rejected-box">
              ❌ LOAN REJECTED<br>
              <span style='font-size:1rem;opacity:0.9;'>Approval probability: {prob[1]*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob[1] * 100,
            title={"text": "Approval Probability (%)", "font": {"size": 18}},
            gauge={
                "axis":  {"range": [0, 100], "tickwidth": 1},
                "bar":   {"color": "#1565c0"},
                "steps": [
                    {"range": [0, 40],  "color": "#ffcdd2"},
                    {"range": [40, 65], "color": "#fff9c4"},
                    {"range": [65, 100],"color": "#c8e6c9"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": 65},
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=60, b=20, l=40, r=40))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Key factors
        st.markdown('<div class="section-title">Key Factors at a Glance</div>', unsafe_allow_html=True)
        factors = {
            "Credit Score": credit_score,
            "DTI Ratio (%)": dti_ratio,
            "Applicant Income": applicant_income,
            "Savings": savings,
        }
        fig_factors = px.bar(
            x=list(factors.values()), y=list(factors.keys()),
            orientation="h",
            color=list(factors.values()),
            color_continuous_scale="Blues",
            title="Applicant Key Metrics",
            template="plotly_white",
        )
        fig_factors.update_layout(coloraxis_showscale=False, height=260,
                                   margin=dict(l=120, r=20, t=50, b=20))
        st.plotly_chart(fig_factors, use_container_width=True)
