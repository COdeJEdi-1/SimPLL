import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Link Spread Insights", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("processed.csv")
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    return df

df = load_data()

# --- Helper Functions ---
def get_domain_data(domain_name):
    if not domain_name:
        return pd.DataFrame()
    return df[df['domain'].str.contains(domain_name, case=False, na=False)]

def plot_time_series(data, label):
    if data.empty:
        return None
    
    ts = data.resample("D", on="created_utc").size()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts.index, ts.values, label=label)
    ax.set_title(f"Posts Over Time")
    ax.set_ylabel("Post Count")
    ax.legend()
    return fig

def plot_top_subreddits(data):
    if data.empty:
        return None
    
    counts = data['subreddit'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title("Top Subreddits Sharing This Domain")
    return fig

# --- Insights Generator ---
def generate_insights(df, domain):
    if df.empty:
        return f"⚠️ No activity detected for **{domain}**."

    ts = df.resample("D", on="created_utc").size()

    recent_avg = ts[-7:].mean() if len(ts) > 7 else ts.mean()
    old_avg = ts[:-7].mean() if len(ts) > 14 else ts.mean()

    trend = (
        "increasing 🔼" if recent_avg > old_avg * 1.3 else
        "declining 🔽" if recent_avg < old_avg * 0.7 else
        "stable ➖"
    )

    spike_day = ts.idxmax().strftime("%b %d %Y")
    spike_value = ts.max()
    top_sub = df['subreddit'].value_counts().idxmax()

    return f"""
### 🧠 Insights for `{domain}`

- Activity trend: **{trend}**
- Peak posting day: **{spike_day}** with **{spike_value} posts**
- Most amplification from: **r/{top_sub}**
- Average posting frequency: **{round(ts.mean(),2)} posts/day**
"""

# --- UI Layout ---
st.title("📊 Link Spread Insights Dashboard")

col1, col2 = st.columns(2)

domain1 = col1.text_input("Domain 1 (e.g., cnn.com):")
domain2 = col2.text_input("Domain 2 (optional):")

filtered1 = get_domain_data(domain1)
filtered2 = get_domain_data(domain2)

# --- Results Count ---
if domain1:
    st.success(f"✅ {len(filtered1)} posts found for {domain1}")

if domain2:
    st.success(f"📌 {len(filtered2)} posts found for {domain2}")

st.write("---")

# --- Summary Stats ---
if domain1:
    st.header("📌 Summary Insights")

    c1, c2, c3 = st.columns(3)

    ts1 = filtered1.resample("D", on="created_utc").size()
    
    c1.metric("Total Posts", len(filtered1))
    c2.metric("Peak Posts in a Day", ts1.max())
    c3.metric("Top Subreddit", filtered1['subreddit'].value_counts().idxmax())

# --- Time Trend Visualization ---
if domain1:
    st.subheader("📈 Trend Over Time")
    fig1 = plot_time_series(filtered1, domain1)
    if fig1:
        st.pyplot(fig1)

# --- Comparative Chart ---
if domain1 and domain2:
    st.subheader("📊 Activity Comparison Chart")

    fig, ax = plt.subplots(figsize=(10,4))

    ts1 = filtered1.resample("D", on="created_utc").size()
    ax.plot(ts1.index, ts1.values, label=domain1)

    if not filtered2.empty:
        ts2 = filtered2.resample("D", on="created_utc").size()
        ax.plot(ts2.index, ts2.values, label=domain2)

    ax.set_title("Posting Activity Comparison")
    ax.set_ylabel("Post Count")
    ax.legend()

    st.pyplot(fig)

# --- Top Subreddits ---
if domain1:
    st.subheader("🔥 Top Subreddits")
    fig_sub1 = plot_top_subreddits(filtered1)
    if fig_sub1:
        st.pyplot(fig_sub1)

# --- Insights Section ---
if domain1:
    st.write("---")
    st.markdown("## 📌 Automated Narrative Insights")
    st.write(generate_insights(filtered1, domain1))

    if domain2:
        st.write(generate_insights(filtered2, domain2))

# Comparison Summary
if domain1 and domain2:
    st.subheader("⚖️ Comparison Verdict")

    diff = len(filtered1) - len(filtered2)
    if diff > 0:
        st.success(f"📢 `{domain1}` appears **more widely circulated** than `{domain2}` by **{diff} posts**.")
    elif diff < 0:
        st.success(f"📢 `{domain2}` is **more amplified** than `{domain1}` by **{abs(diff)} posts**.")
    else:
        st.info("⚖️ Both domains show equal visibility.")

