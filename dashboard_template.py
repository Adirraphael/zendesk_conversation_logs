import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Wattly AI — QA Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ── FILE UPLOAD OR PATH ───────────────────────────────────────────────────────
st.title("🤖 Wattly AI Chatbot — QA Performance Dashboard")

uploaded = st.file_uploader("Upload your results CSV", type=["csv"])

if not uploaded:
    st.info("👆 Upload a graded results CSV to get started.")
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
    for col in ["accuracy_helpfulness", "retrieval_quality", "conversational_ux",
                 "safety_compliance", "business_outcomes", "success_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data(uploaded)
st.caption(f"{len(df)} questions loaded")
st.divider()

# ── TOP KPI ROW ───────────────────────────────────────────────────────────────
total      = len(df)
responded  = (df["responded"] == "YES").sum() if "responded" in df.columns else total
outliers   = (df["outlier"] == "YES").sum() if "outlier" in df.columns else 0
avg_score  = df["success_rate"].mean() if "success_rate" in df.columns else 0
respond_rt = round(responded / total * 100, 1) if total else 0
outlier_rt = round(outliers / total * 100, 1) if total else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📋 Total Questions", total)
k2.metric("✅ Response Rate", f"{respond_rt}%", f"{responded}/{total} responded")
k3.metric("⭐ Avg Success Rate", f"{avg_score:.1f}%")
k4.metric("⚠️ Outliers", outliers, f"{outlier_rt}% of total", delta_color="inverse")
k5.metric("❌ No Response", total - responded)

st.divider()

# ── ROW 1 ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Success Rate Distribution")
    fig = px.histogram(df, x="success_rate", nbins=20,
                       color_discrete_sequence=["#2563EB"],
                       labels={"success_rate": "Success Rate (%)"})
    fig.add_vline(x=avg_score, line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {avg_score:.1f}%")
    fig.update_layout(bargap=0.1, showlegend=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏷️ Questions by Category")
    if "category" in df.columns:
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig2 = px.pie(cat_counts, names="Category", values="Count",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── ROW 2 ─────────────────────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Avg Success Rate by Category")
    if "category" in df.columns and "success_rate" in df.columns:
        cat_avg = df.groupby("category")["success_rate"].mean().sort_values().reset_index()
        cat_avg.columns = ["Category", "Avg Success Rate"]
        fig3 = px.bar(cat_avg, x="Avg Success Rate", y="Category", orientation="h",
                      color="Avg Success Rate", color_continuous_scale="Blues",
                      text=cat_avg["Avg Success Rate"].apply(lambda x: f"{x:.1f}%"))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("🎯 Average Score by Criteria")
    criteria = {
        "Accuracy &\nHelpfulness": df["accuracy_helpfulness"].mean() if "accuracy_helpfulness" in df.columns else 0,
        "Retrieval\nQuality":      df["retrieval_quality"].mean() if "retrieval_quality" in df.columns else 0,
        "Conversational\nUX":      df["conversational_ux"].mean() if "conversational_ux" in df.columns else 0,
        "Safety &\nCompliance":    df["safety_compliance"].mean() if "safety_compliance" in df.columns else 0,
        "Business\nOutcomes":      df["business_outcomes"].mean() if "business_outcomes" in df.columns else 0,
    }
    fig4 = go.Figure(go.Bar(
        x=list(criteria.keys()),
        y=list(criteria.values()),
        marker_color=["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED"],
        text=[f"{v:.2f}" for v in criteria.values()],
        textposition="outside"
    ))
    fig4.update_layout(yaxis=dict(range=[0, 5], title="Score (1–5)"),
                       plot_bgcolor="white", paper_bgcolor="white",
                       showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# ── ROW 3 ─────────────────────────────────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("⚠️ Outlier Breakdown")
    if "outlier" in df.columns:
        out_counts = df["outlier"].value_counts().reset_index()
        out_counts.columns = ["Outlier", "Count"]
        fig5 = px.pie(out_counts, names="Outlier", values="Count",
                      color="Outlier",
                      color_discrete_map={"YES": "#DC2626", "NO": "#16A34A"})
        fig5.update_traces(textinfo="percent+value")
        st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("✅ Response Status")
    if "responded" in df.columns:
        res_counts = df["responded"].value_counts().reset_index()
        res_counts.columns = ["Responded", "Count"]
        fig6 = px.pie(res_counts, names="Responded", values="Count",
                      color="Responded",
                      color_discrete_map={"YES": "#2563EB", "NO": "#DC2626"})
        fig6.update_traces(textinfo="percent+value")
        st.plotly_chart(fig6, use_container_width=True)

# ── ROW 4 ─────────────────────────────────────────────────────────────────────
if "category" in df.columns and "outlier" in df.columns:
    st.subheader("🔴 Outlier Rate by Category")
    cat_outlier = df.groupby("category").apply(
        lambda x: round((x["outlier"] == "YES").sum() / len(x) * 100, 1)
    ).reset_index()
    cat_outlier.columns = ["Category", "Outlier Rate (%)"]
    cat_outlier = cat_outlier.sort_values("Outlier Rate (%)", ascending=False)
    fig7 = px.bar(cat_outlier, x="Category", y="Outlier Rate (%)",
                  color="Outlier Rate (%)", color_continuous_scale="Reds",
                  text=cat_outlier["Outlier Rate (%)"].apply(lambda x: f"{x}%"))
    fig7.update_traces(textposition="outside")
    fig7.update_layout(coloraxis_showscale=False,
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig7, use_container_width=True)

# ── TABLE ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📄 Full Results Table")

fc1, fc2, fc3 = st.columns(3)
with fc1:
    if "category" in df.columns:
        cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Filter by Category", cats)
    else:
        sel_cat = "All"
with fc2:
    if "outlier" in df.columns:
        sel_outlier = st.selectbox("Filter by Outlier", ["All", "YES", "NO"])
    else:
        sel_outlier = "All"
with fc3:
    if "responded" in df.columns:
        sel_responded = st.selectbox("Filter by Responded", ["All", "YES", "NO"])
    else:
        sel_responded = "All"

filtered = df.copy()
if "category" in df.columns and sel_cat != "All":
    filtered = filtered[filtered["category"] == sel_cat]
if "outlier" in df.columns and sel_outlier != "All":
    filtered = filtered[filtered["outlier"] == sel_outlier]
if "responded" in df.columns and sel_responded != "All":
    filtered = filtered[filtered["responded"] == sel_responded]

st.dataframe(filtered, use_container_width=True, height=400)
st.caption(f"Showing {len(filtered)} of {total} rows")