from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from azure_language_helper import extract_key_phrases_from_azure

st.set_page_config(page_title="InclusionPulse AI", layout="wide")

DATA_PATH = Path("data/processed/final_dataset.csv")

FEATURE_COLS = [
    "internet_users_pct",
    "mobile_subscriptions_per_100",
    "bank_branches_per_100k_adults",
    "atms_per_100k_adults",
    "private_credit_pct_gdp",
    "gdp_growth_pct",
    "inflation_pct",
    "unemployment_pct",
]

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(df):
    if df is None or len(df) < 8:
        return None, None

    X = df[FEATURE_COLS].copy()
    y = df["intervention_priority_score"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
    }
    return model, metrics

def generate_policy_brief(latest_row: pd.Series) -> str:
    return (
        f"Pada tahun {int(latest_row['year'])}, Indonesia memiliki intervention priority score sebesar "
        f"{latest_row['intervention_priority_score']:.2f} dengan priority tier {latest_row['priority_tier']}. "
        f"Pilar terlemah saat ini adalah {latest_row['weakest_pillar']}. "
        f"Temuan ini menunjukkan bahwa intervensi nasional sebaiknya difokuskan terlebih dahulu pada pilar tersebut "
        f"untuk memperkuat ketahanan ekonomi digital dan memperluas inklusi keuangan secara lebih efektif."
    )

st.title("InclusionPulse AI")
st.caption("National Intervention Prioritization for Digital Economy & Financial Inclusion")

df = load_data()
if df is None:
    st.error("Dataset belum ada. Jalankan GitHub Action atau script pipeline terlebih dahulu.")
    st.stop()

model, metrics = train_model(df)

latest = df.sort_values("year").iloc[-1]
brief = generate_policy_brief(latest)

col1, col2, col3 = st.columns(3)
col1.metric("Latest Year", int(latest["year"]))
col2.metric("Priority Score", f"{latest['intervention_priority_score']:.2f}")
col3.metric("Priority Tier", latest["priority_tier"])

tab1, tab2, tab3, tab4 = st.tabs([
    "Trend",
    "Dataset",
    "Model",
    "Policy Brief + Azure"
])

with tab1:
    st.subheader("Intervention Priority Trend")
    st.line_chart(df.set_index("year")[["intervention_priority_score", "overall_resilience_score"]])

    st.subheader("Pillar Trend")
    st.line_chart(df.set_index("year")[[
        "digital_access_index",
        "financial_inclusion_index",
        "macro_resilience_index"
    ]])

with tab2:
    st.subheader("Final Dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download final dataset",
        df.to_csv(index=False).encode("utf-8"),
        file_name="final_dataset.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("AI Model")
    if model is None:
        st.info("Data masih terlalu sedikit untuk model yang stabil.")
    else:
        st.write(f"MAE: {metrics['mae']}")
        st.write(f"R²: {metrics['r2']}")

with tab4:
    st.subheader("Policy Brief")
    st.text_area("Generated brief", brief, height=220)

    azure_result = extract_key_phrases_from_azure(brief)

    if azure_result["status"] == "ok":
        st.success(azure_result["message"])
        st.write("Key phrases from Azure AI Language:")
        st.write(", ".join(azure_result["key_phrases"]))
    elif azure_result["status"] == "not_configured":
        st.info("Azure belum dikonfigurasi. App tetap berjalan tanpa Azure.")
    else:
        st.warning(f"Azure error: {azure_result['message']}")
