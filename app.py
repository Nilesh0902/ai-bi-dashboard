import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(layout="wide")

st.title("🚀 AI Powered Business Intelligence Dashboard")

st.markdown("""
### 📊 Integrated Business Intelligence + Machine Learning Platform  
This dashboard combines descriptive analytics, predictive modeling,  
customer segmentation, and anomaly detection in one interactive system.
""")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("superstore.csv.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    return df

df = load_data()

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("🔎 Filters")

region = st.sidebar.multiselect(
    "Select Region",
    df['Region'].unique(),
    default=df['Region'].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    df['Category'].unique(),
    default=df['Category'].unique()
)

filtered_df = df[
    (df['Region'].isin(region)) &
    (df['Category'].isin(category))
]

# ---------------------------------------------------
# CREATE TABS
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Business Overview", "🤖 Machine Learning", "📈 Advanced Insights"]
)

# ===================================================
# TAB 1 — BUSINESS OVERVIEW
# ===================================================
with tab1:

    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    profit_margin = (total_profit / total_sales) * 100 if total_sales != 0 else 0
    total_orders = filtered_df.shape[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📈 Total Profit", f"${total_profit:,.0f}")
    col3.metric("📊 Profit Margin", f"{profit_margin:.2f}%")
    col4.metric("🧾 Total Orders", total_orders)

    st.markdown("---")

    # Sales Trend
    st.subheader("📈 Sales Trend Over Time")

    sales_trend = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()

    fig = px.line(
        sales_trend,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Sales Trend Over Time"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    if len(sales_trend) > 1:
        growth = sales_trend['Sales'].pct_change().mean() * 100
        st.info(f"📌 Average Growth Rate: {growth:.2f}% per period")

    st.markdown("---")

    # Region Analysis
    st.subheader("🌍 Sales by Region")

    region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()

    fig2 = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Region",
        title="Regional Sales Distribution"
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)


# ===================================================
# TAB 2 — MACHINE LEARNING
# ===================================================
with tab2:

    st.subheader("👥 Customer Segmentation (KMeans)")

    rfm = filtered_df.groupby("Customer ID").agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()

    if len(rfm) > 3:
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm[['Sales', 'Profit']])

        fig3 = px.scatter(
            rfm,
            x="Sales",
            y="Profit",
            color=rfm['Cluster'].astype(str),
            title="Customer Clusters"
        )
        fig3.update_layout(template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Not enough data for clustering.")

    st.markdown("---")

    st.subheader("⚠️ Anomaly Detection (Isolation Forest)")

    if len(filtered_df) > 10:
        sales_trend = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()

        iso = IsolationForest(contamination=0.02, random_state=42)
        sales_trend['Anomaly'] = iso.fit_predict(sales_trend[['Sales']])

        fig4 = px.scatter(
            sales_trend,
            x="Order Date",
            y="Sales",
            color=sales_trend['Anomaly'].map({1: "Normal", -1: "Anomaly"}),
            title="Anomaly Detection in Sales"
        )
        fig4.update_layout(template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Not enough data for anomaly detection.")

    st.markdown("---")

    st.subheader("🔮 Sales Prediction (Random Forest)")

    ml_df = filtered_df[['Year', 'Month', 'Sales']]

    if len(ml_df) > 10:
        X = ml_df[['Year', 'Month']]
        y = ml_df['Sales']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        r2 = model.score(X_test, y_test)

        st.success(f"Model R² Score: {r2:.4f}")
    else:
        st.warning("Not enough data for prediction model.")


# ===================================================
# TAB 3 — ADVANCED INSIGHTS
# ===================================================
with tab3:

    st.subheader("📊 Business Summary Insights")

    if total_profit > 0:
        st.success("Overall business is profitable.")
    else:
        st.error("Business is running at a loss.")

    if profit_margin < 10:
        st.warning("Profit margin is relatively low. Cost optimization recommended.")

    if total_sales > 100000:
        st.info("High revenue performance observed.")

    st.markdown("---")
    st.markdown("### 👨‍💻 Developed by Nilesh Singh")
    st.markdown("MSc Data Science | AI Powered BI Dashboard")