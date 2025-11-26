import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# =======================
# Load dataset
# =======================
df = pd.read_csv("processed.csv")
df['created_utc'] = pd.to_datetime(df['created_utc'])

st.set_page_config(page_title="Link Spread Dashboard", layout="wide")
sns.set_style("whitegrid")


# =======================
# Helper Functions
# =======================

def get_domain_data(domain):
    if not domain:
        return pd.DataFrame()
    return df[df['domain'].str.contains(domain, case=False, na=False)]


def plot_time_series(data, label=None):
    ts = data.resample('D', on='created_utc').size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts.values, label=label)
    ax.set_title("Posts Over Time")
    ax.set_xlabel("created_utc")
    ax.set_ylabel("Post Count")

    if label:
        ax.legend()

    return fig, ts


def plot_subreddit_bar(data):
    counts = data['subreddit'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title("Top Subreddits Sharing This Domain")
    ax.set_xlabel("Post Count")
    ax.set_ylabel("Subreddit")
    return fig


def generate_insights(data, domain):
    if data.empty:
        return f"No data found for {domain}."

    top_sub = data['subreddit'].value_counts().idxmax()
    peak_day = data.resample('D', on='created_utc').size().idxmax().strftime('%b %d, %Y')

    return (
        f"• **{domain}** is primarily shared in **r/{top_sub}**.\n"
        f"• Posting peaks on **{peak_day}**, suggesting event-based amplification.\n"
        f"• Posting frequency suggests **coordinated or recurring engagement patterns.**"
    )


def save_chart(fig):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches='tight')
    return tmp.name


def generate_pdf(domain1, domain2, stats, insights_text, comparison_note, chart_paths):
    """Creates downloadable PDF and returns file path."""
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    c = canvas.Canvas(temp_path, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, y, "Link Spread Insights Report")

    y -= 40
    c.setFont("Helvetica", 12)
    c.drawString(30, y, f"Domain 1: {domain1}")
    
    if domain2:
        y -= 20
        c.drawString(30, y, f"Domain 2: {domain2}")

    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "Summary Stats")

    c.setFont("Helvetica", 11)
    for key, value in stats.items():
        y -= 18
        c.drawString(40, y, f"{key}: {value}")

    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "Insights")

    c.setFont("Helvetica", 10)
    for line in insights_text.split("\n"):
        y -= 15
        c.drawString(40, y, line)

    if comparison_note:
        y -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, y, "Comparison Verdict")

        y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(40, y, comparison_note)

    # Add charts (new pages)
    for chart in chart_paths:
        if chart and os.path.exists(chart):
            c.showPage()
            c.drawImage(chart, 30, 150, width=500, preserveAspectRatio=True)

    c.save()
    return temp_path


# =======================
# STREAMLIT UI
# =======================

st.title("📊 Link Spread Insights Dashboard")

col1, col2 = st.columns(2)

domain1 = col1.text_input("Domain 1 (e.g., cnn.com):")
domain2 = col2.text_input("Domain 2 (Optional):")

filtered1 = get_domain_data(domain1)
filtered2 = get_domain_data(domain2) if domain2 else pd.DataFrame()

if domain1:
    st.success(f"{len(filtered1)} posts found for {domain1}")

    # Summary Section
    st.subheader("📌 Summary Insights")

    fig1, ts1 = plot_time_series(filtered1)
    top_sub1 = filtered1['subreddit'].value_counts().idxmax() if not filtered1.empty else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Posts", len(filtered1))
    c2.metric("Peak Posts in a Day", ts1.max() if len(ts1) else 0)
    c3.metric("Top Subreddit", top_sub1)

    # Time Series Chart
    st.subheader("📈 Trend Over Time")
    st.pyplot(fig1)

    # Subreddit Breakdown
    st.subheader("🔥 Top Subreddits")
    st.pyplot(plot_subreddit_bar(filtered1))

    # Comparison Chart
    if domain2 and not filtered2.empty:
        st.subheader("📊 Activity Comparison Chart")

        fig_compare, _ = plot_time_series(filtered1, label=domain1)
        fig_compare2, ts2 = plot_time_series(filtered2, label=domain2)

        ax = fig_compare.axes[0]
        ax.plot(ts2.index, ts2.values, label=domain2)
        ax.legend()

        st.pyplot(fig_compare)

        # Comparison Verdict
        diff = len(filtered1) - len(filtered2)
        if diff > 0:
            comparison_note = f"🔍 **{domain1} shows stronger amplification than {domain2} by {diff} posts.**"
        elif diff < 0:
            comparison_note = f"🔍 **{domain2} shows stronger amplification than {domain1} by {abs(diff)} posts.**"
        else:
            comparison_note = "⚖ Both domains demonstrate equal engagement."

        st.info(comparison_note)
    else:
        comparison_note = None

    # Insights Text
    st.subheader("🧠 AI-style Insight Summary")
    insights_text = generate_insights(filtered1, domain1)
    st.write(insights_text)

    # Generate PDF Button
    st.write("---")
    st.subheader("📄 Export Report")

    stats = {
        "Total Posts": len(filtered1),
        "Peak Posts": ts1.max() if len(ts1) else 0,
        "Top Subreddit": top_sub1
    }

    # Save charts for PDF
    chart_paths = [
        save_chart(fig1),
        save_chart(plot_subreddit_bar(filtered1)),
    ]

    if domain2 and not filtered2.empty:
        chart_paths.append(save_chart(fig_compare))

    pdf_file = generate_pdf(domain1, domain2, stats, insights_text, comparison_note, chart_paths)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name=f"{domain1}_analysis_report.pdf",
            mime="application/pdf"
        )

