import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise | Loan Approval System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0D1117;
    color: #E6EDF3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] * {
    color: #C9D1D9 !important;
}

/* Header strip */
.cw-header {
    background: linear-gradient(135deg, #1C2733 0%, #0D1117 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.cw-logo {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #58A6FF;
}
.cw-logo span { color: #3FB950; }
.cw-tagline {
    font-size: 0.85rem;
    color: #8B949E;
    margin-top: 2px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #58A6FF;
    font-family: 'DM Mono', monospace;
}

/* Result badge */
.result-approved {
    background: linear-gradient(135deg, #1B4332, #0F2A1D);
    border: 1.5px solid #3FB950;
    border-radius: 14px;
    padding: 28px 36px;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, #3B1219, #1F0A0E);
    border: 1.5px solid #F85149;
    border-radius: 14px;
    padding: 28px 36px;
    text-align: center;
}
.result-title {
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.result-sub {
    font-size: 0.9rem;
    color: #8B949E;
}

/* Section titles */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #58A6FF;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262D;
}

/* Probability bar */
.prob-bar-wrap {
    background: #21262D;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin-top: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* Tabs override */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8B949E !important;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #E6EDF3 !important;
}

/* Input labels */
label { color: #C9D1D9 !important; font-size: 0.85rem !important; }

/* Buttons */
.stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2EA043; }

/* divider */
hr { border-color: #21262D; }

/* Selectbox and number inputs */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput > div > div {
    background: #21262D !important;
    border-color: #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on loan data…")
def load_model_and_artifacts(csv_path: str):
    df = pd.read_csv(csv_path)

    # --- Impute ---
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols   = df.select_dtypes(include=["number"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # --- Label encode ---
    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    le_target = LabelEncoder()
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

    # --- Drop ID ---
    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)

    # --- OHE ---
    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # --- Feature engineering ---
    df["DTI_Ratio_sq"]   = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved","Credit_Score","DTI_Ratio"])
    y = df["Loan_Approved"]

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred) * 100, 2),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
    }

    return model, scaler, ohe, le_edu, le_target, feature_cols, metrics, df


def build_input_vector(inputs: dict, ohe, le_edu, feature_cols):
    """Turn form inputs into a 1-row numpy array matching training features."""
    # OHE
    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    raw_ohe = pd.DataFrame([[inputs[c] for c in ohe_cols]], columns=ohe_cols)
    encoded = ohe.transform(raw_ohe)
    ohe_df  = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))

    # Numeric base
    edu_encoded = le_edu.transform([inputs["Education_Level"]])[0]
    base = {
        "Age":              inputs["Age"],
        "Applicant_Income": inputs["Applicant_Income"],
        "Co_Applicant_Income": inputs["Co_Applicant_Income"],
        "Loan_Amount":      inputs["Loan_Amount"],
        "Loan_Term":        inputs["Loan_Term"],
        "Credit_Score_sq":  inputs["Credit_Score"] ** 2,
        "DTI_Ratio_sq":     inputs["DTI_Ratio"] ** 2,
        "Savings":          inputs["Savings"],
        "Existing_Loan_Balance": inputs["Existing_Loan_Balance"],
        "Number_of_Dependents":  inputs["Number_of_Dependents"],
        "Education_Level":  edu_encoded,
    }
    base_df = pd.DataFrame([base])

    combined = pd.concat([base_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    # Align to training columns
    for col in feature_cols:
        if col not in combined.columns:
            combined[col] = 0
    combined = combined[feature_cols]
    return combined.values


# ─────────────────────────────────────────────
# SIDEBAR – data upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload Training Data (CSV)",
        type=["csv"],
        help="Upload your loan_approval_data.csv to train the model."
    )
    if uploaded:
        save_path = f"/tmp/{uploaded.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state["csv_path"] = save_path
        st.success(f"✅ Loaded: {uploaded.name}")
    elif "csv_path" not in st.session_state:
        st.info("Upload your dataset to get started.")

    st.markdown("---")
    st.markdown("**About**")
    st.caption("CreditWise uses Naive Bayes classification trained on your loan portfolio data to predict approval likelihood.")
    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="cw-header">
  <div>
    <div class="cw-logo">Credit<span>Wise</span></div>
    <div class="cw-tagline">AI-Powered Loan Approval System</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Predict Approval", "📊  Model Performance", "📋  About"])

# ══════════════════════════════════════════════
# TAB 1 – PREDICT
# ══════════════════════════════════════════════
with tab1:
    if "csv_path" not in st.session_state:
        st.warning("👈 Please upload your loan dataset from the sidebar to enable predictions.")
    else:
        try:
            model, scaler, ohe, le_edu, le_target, feature_cols, metrics, df_trained = \
                load_model_and_artifacts(st.session_state["csv_path"])
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # ── FORM ──────────────────────────────
        st.markdown('<div class="section-title">Applicant Information</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Details**")
            age    = st.number_input("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            dependents = st.number_input("Number of Dependents", 0, 10, 0)
            edu    = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])

        with col2:
            st.markdown("**Financial Details**")
            income    = st.number_input("Applicant Income (₹)", 0, 10_000_000, 50_000, step=1000)
            co_income = st.number_input("Co-Applicant Income (₹)", 0, 5_000_000, 0, step=1000)
            savings   = st.number_input("Savings (₹)", 0, 10_000_000, 100_000, step=5000)
            existing  = st.number_input("Existing Loan Balance (₹)", 0, 5_000_000, 0, step=5000)
            credit    = st.number_input("Credit Score", 300, 900, 700)

        with col3:
            st.markdown("**Loan Details**")
            loan_amt  = st.number_input("Loan Amount (₹)", 10_000, 50_000_000, 500_000, step=10_000)
            loan_term = st.number_input("Loan Term (months)", 6, 360, 120)
            dti       = st.number_input("DTI Ratio (%)", 0.0, 100.0, 30.0, step=0.5)
            emp_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Retired"])
            emp_cat    = st.selectbox("Employer Category", ["Government", "Private", "NGO", "Self"])
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business", "Medical", "Other"])
            property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

        st.markdown("---")
        predict_btn = st.button("⚡ Run Credit Assessment", use_container_width=True)

        if predict_btn:
            inputs = {
                "Age": age, "Gender": gender, "Marital_Status": marital,
                "Number_of_Dependents": dependents, "Education_Level": edu,
                "Applicant_Income": income, "Co_Applicant_Income": co_income,
                "Savings": savings, "Existing_Loan_Balance": existing,
                "Credit_Score": credit, "Loan_Amount": loan_amt,
                "Loan_Term": loan_term, "DTI_Ratio": dti,
                "Employment_Status": emp_status, "Employer_Category": emp_cat,
                "Loan_Purpose": loan_purpose, "Property_Area": property_area,
            }

            try:
                X_input = build_input_vector(inputs, ohe, le_edu, feature_cols)
                X_scaled = scaler.transform(X_input)
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                approved_prob = round(float(proba[1]) * 100, 1)
                rejected_prob = round(float(proba[0]) * 100, 1)

                st.markdown("---")
                r1, r2 = st.columns([2, 1])

                with r1:
                    if pred == 1:
                        st.markdown(f"""
                        <div class="result-approved">
                          <div class="result-title" style="color:#3FB950">✅ Loan Approved</div>
                          <div class="result-sub">The applicant meets the credit criteria. Approval confidence: <b>{approved_prob}%</b></div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-rejected">
                          <div class="result-title" style="color:#F85149">❌ Loan Rejected</div>
                          <div class="result-sub">The applicant does not meet credit criteria. Rejection confidence: <b>{rejected_prob}%</b></div>
                        </div>""", unsafe_allow_html=True)

                with r2:
                    st.markdown('<div class="metric-card"><div class="label">Approval Probability</div>'
                                f'<div class="value">{approved_prob}%</div></div>', unsafe_allow_html=True)
                    bar_color = "#3FB950" if approved_prob >= 50 else "#F85149"
                    st.markdown(f"""
                    <div class="prob-bar-wrap">
                      <div class="prob-bar-fill" style="width:{approved_prob}%; background:{bar_color};"></div>
                    </div>""", unsafe_allow_html=True)

                # Quick factor summary
                st.markdown("---")
                st.markdown('<div class="section-title">Key Factors Assessed</div>', unsafe_allow_html=True)
                f1c, f2c, f3c, f4c = st.columns(4)
                f1c.metric("Credit Score",    f"{credit}",         delta="Good" if credit >= 650 else "Low")
                f2c.metric("DTI Ratio",       f"{dti}%",           delta="Healthy" if dti < 40 else "High", delta_color="inverse")
                f3c.metric("Savings",         f"₹{savings:,.0f}")
                f4c.metric("Loan-to-Income",  f"{loan_amt/max(income,1):.1f}x", delta="OK" if loan_amt/max(income,1) < 5 else "High", delta_color="inverse")

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════
# TAB 2 – PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    if "csv_path" not in st.session_state:
        st.warning("👈 Upload the dataset first.")
    else:
        try:
            model, scaler, ohe, le_edu, le_target, feature_cols, metrics, df_trained = \
                load_model_and_artifacts(st.session_state["csv_path"])
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        st.markdown('<div class="section-title">Naive Bayes Model Metrics</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col, label, key in zip(
            [m1, m2, m3, m4],
            ["Accuracy", "Precision", "Recall", "F1 Score"],
            ["accuracy", "precision", "recall", "f1"]
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{metrics[key]}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Confusion matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(metrics["cm"])
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#161B22")
        ax.set_facecolor("#161B22")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Rejected","Approved"],
                    yticklabels=["Rejected","Approved"], ax=ax,
                    annot_kws={"color":"white","size":13})
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values(): spine.set_edgecolor("#21262D")
        ax.set_xlabel("Predicted", color="#8B949E")
        ax.set_ylabel("Actual", color="#8B949E")
        ax.set_title("Confusion Matrix", color="#58A6FF", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown('<div class="section-title">Model Details</div>', unsafe_allow_html=True)
        st.info("**Algorithm:** Gaussian Naive Bayes  |  **Features:** 30+ engineered variables  |  **Split:** 80/20 train-test  |  **Feature Engineering:** DTI² & CreditScore² added for non-linearity")


# ══════════════════════════════════════════════
# TAB 3 – ABOUT
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    ## About CreditWise

    **CreditWise** is a machine learning–powered loan approval prediction system built with:
    - **Scikit-learn** – Naive Bayes classification, preprocessing, and evaluation
    - **Streamlit** – Interactive web application
    - **Pandas / NumPy** – Data manipulation and feature engineering

    ### How It Works
    1. **Data Preprocessing** — Missing values are imputed (mean for numerics, mode for categoricals)
    2. **Encoding** — Education level is label-encoded; categorical variables are one-hot encoded
    3. **Feature Engineering** — DTI Ratio² and Credit Score² are added to capture non-linear patterns
    4. **Scaling** — StandardScaler normalises all features for optimal Naive Bayes performance
    5. **Model** — Gaussian Naive Bayes is trained and evaluated on an 80/20 split

    ### Input Variables
    | Variable | Type | Description |
    |---|---|---|
    | Age | Numeric | Applicant's age |
    | Gender | Categorical | Gender identity |
    | Marital Status | Categorical | Marital situation |
    | Education Level | Ordinal | Highest qualification |
    | Applicant Income | Numeric | Monthly income |
    | Co-Applicant Income | Numeric | Co-applicant's income |
    | Loan Amount | Numeric | Requested loan amount |
    | Loan Term | Numeric | Repayment period in months |
    | Credit Score | Numeric | Credit bureau score (300–900) |
    | DTI Ratio | Numeric | Debt-to-income ratio |
    | Savings | Numeric | Total savings |
    | Employment Status | Categorical | Current employment |
    | Employer Category | Categorical | Type of employer |
    | Loan Purpose | Categorical | Intended use of funds |
    | Property Area | Categorical | Urban / Semi-Urban / Rural |

    ---
    *Built as part of the CreditWise Loan System project.*
    """)
