import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("processed.csv")
df["created_utc"] = pd.to_datetime(df["created_utc"])  # required for resampling

domain_df = df.copy()

def get_domain_data(domain_name):
    return domain_df[domain_df['domain'].str.contains(domain_name, case=False, na=False)]

def plot_time_series(data):
    ts = data.resample('D', on='created_utc').size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax)
    ax.set_title("Posts Over Time")
    return fig

def plot_subreddit_bar(data):
    counts = data['subreddit'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title("Top Subreddits Sharing This Domain")
    ax.set_xlabel("Post Count")
    ax.set_ylabel("Subreddit")
    return fig


st.title("📊 Link Spread Insights Dashboard")

domain_input = st.text_input("Enter a domain (e.g., cnn.com):")

if domain_input:
    filtered = get_domain_data(domain_input)

    if filtered.empty:
        st.warning("No results found for this domain.")
    else:
        # --- Basic count info ---
        st.success(f"Results found: {len(filtered)}")

        # --- Summary insights ---
        st.subheader("Summary Insights")

        # daily activity
        daily_counts = filtered.resample("D", on="created_utc").size()
        peak_day = daily_counts.idxmax().strftime("%b %d, %Y")
        peak_value = int(daily_counts.max())
        avg_daily = round(daily_counts.mean(), 2)

        # top subreddit
        top_subreddit = filtered["subreddit"].value_counts().idxmax()

        col1, col2, col3 = st.columns(3)
        col1.metric("Peak posts in a day", peak_value, help="Highest number of posts in a single day")
        col2.metric("Average posts per day", avg_daily)
        col3.metric("Top subreddit", top_subreddit)

        # --- Plots ---
        st.write("### Trend Over Time")
        st.pyplot(plot_time_series(filtered))

        st.write("### Top Subreddits")
        st.pyplot(plot_subreddit_bar(filtered))

        st.pyplot(plot_subreddit_bar(filtered))
