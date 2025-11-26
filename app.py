import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvis.network import Network
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("processed.csv")
df['created_utc'] = pd.to_datetime(df['created_utc'])

# -------------------------------
# Load NLP Models
# -------------------------------
@st.cache_resource
def load_models():
    sentiment_model = SentimentIntensityAnalyzer()
    toxicity_pipe = pipeline("text-classification", model="unitary/toxic-bert", truncation=True)
    emotion_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", top_k=1)
    return sentiment_model, toxicity_pipe, emotion_pipe

sentiment_model, toxicity_pipe, emotion_pipe = load_models()

# -------------------------------
# Compute NLP Features
# -------------------------------
@st.cache_data
def compute_text_features(data):
    data = data.copy()

    # sentiment
    data["sentiment"] = data["title"].apply(lambda x: sentiment_model.polarity_scores(str(x))['compound'])

    # toxicity
    data["toxicity"] = data["title"].apply(lambda x: toxicity_pipe(str(x))[0]['score'])

    # emotion classification
    def safe_emotion(text):
        try:
            return emotion_pipe(str(text))[0][0]['label']
        except:
            return "unknown"

    data["emotion"] = data["title"].apply(safe_emotion)

    return data

# -------------------------------
# Load Chatbot Components
# -------------------------------
@st.cache_resource
def load_chat_components():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", max_new_tokens=300)
    return embedder, qa_model

embedder, qa_model = load_chat_components()

# -------------------------------
# UI
# -------------------------------
st.title("📊 Link Spread Intelligence Dashboard")

domain1 = st.text_input("Enter a news domain (e.g., cnn.com):")
domain2 = st.text_input("Optional second domain for comparison:")

if domain1:

    filtered = df[df['domain'].str.contains(domain1, case=False, na=False)]
    filtered = compute_text_features(filtered)

    st.success(f"{len(filtered)} posts found for {domain1}")

    # -------------------------------
    # Summary Metrics
    # -------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", len(filtered))
    col2.metric("Avg Sentiment", round(filtered['sentiment'].mean(), 3))
    col3.metric("Avg Toxicity", round(filtered['toxicity'].mean(), 3))

    # -------------------------------
    # Emotion Chart
    # -------------------------------
    st.subheader("🧠 Emotion Distribution")
    emotion_counts = filtered['emotion'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=emotion_counts.values, y=emotion_counts.index, ax=ax)
    st.pyplot(fig)

    # -------------------------------
    # Time Series Trend
    # -------------------------------
    st.subheader("📈 Trend Over Time")
    ts = filtered.resample('D', on='created_utc').size()
    fig2, ax2 = plt.subplots()
    ts.plot(ax=ax2)
    st.pyplot(fig2)

    # -------------------------------
    # Top Subreddits
    # -------------------------------
    st.subheader("🔥 Top Subreddits Posting This Domain")
    sr = filtered['subreddit'].value_counts().head(10)
    fig3, ax3 = plt.subplots()
    sns.barplot(x=sr.values, y=sr.index, ax=ax3)
    st.pyplot(fig3)

    # -------------------------------
    # Network Graph
    # -------------------------------
    st.subheader("🕸 Network Graph: Spread Pattern")

    def create_network_graph(df, domain):
        net = Network(height="550px", width="100%", bgcolor="#111", font_color="white")
        net.add_node(domain, label=domain, color="#00c3ff", size=25)

        subreddit_counts = df['subreddit'].value_counts().head(15)
        for subreddit, count in subreddit_counts.items():
            net.add_node(subreddit, label=subreddit, size=15 + (count / 3), color="#ff6b6b")
            net.add_edge(domain, subreddit, value=count)

        net.force_atlas_2based()
        return net

    if not filtered.empty:
        net = create_network_graph(filtered, domain1)
        path = "network_graph.html"
        net.save_graph(path)
        components.html(open(path, "r", encoding="utf-8").read(), height=600)

    # -------------------------------
    # Topic Clustering
    # -------------------------------
    st.subheader("🧩 Topic Clustering (Auto Themes)")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(filtered['title'])
    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered['cluster'] = kmeans.fit_predict(X)
    st.write(filtered[['title', 'cluster']].head(20))

    # -------------------------------
    # Story Summary
    # -------------------------------
    st.subheader("📌 AI Summary Report")

    def generate_story(data, dom):
        tone = "negative" if data['sentiment'].mean() < -0.1 else "mixed" if data['sentiment'].mean() < 0.1 else "positive"
        top_emotion = data['emotion'].value_counts().idxmax()
        top_sub = data['subreddit'].value_counts().idxmax()

        return f"""
        The domain **{dom}** appears **{len(data)} times** in Reddit conversations.
        Tone is **{tone}**, most emotionally represented by **{top_emotion}**.
        Most activity originates from **r/{top_sub}**.

        Peak posting activity reached **{data.resample('D', on='created_utc').size().max()} posts/day**, indicating event-driven spikes.

        Content forms **{data['cluster'].nunique()} main thematic clusters**, revealing diverse narratives.
        """

    st.info(generate_story(filtered, domain1))

    # -------------------------------
    # Chatbot Assistant
    # -------------------------------
    st.subheader("💬 Ask the Dataset Anything")

    if len(filtered) > 5:

        @st.cache_data
        def build_index(df):
            docs = df["title"].tolist()
            vectors = embedder.encode(docs)
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            return index, docs

        index, docs = build_index(filtered)

        question = st.text_input("Ask a question (e.g., 'Why is this domain increasing in Feb?'):")

        if question:
            q_vec = embedder.encode([question])
            scores, result_ids = index.search(q_vec, k=5)
            matched_texts = "\n".join([f"- {docs[i]}" for i in result_ids[0]])

            prompt = f"""
            You are a Reddit trend analyst. Based on these posts and analytics below,
            answer the question clearly and concisely.

            Question:
            {question}

            Relevant Posts:
            {matched_texts}

            Stats:
            - Total posts: {len(filtered)}
            - Avg sentiment: {filtered['sentiment'].mean():.2f}
            - Dominant emotion: {filtered['emotion'].value_counts().idxmax()}
            - Key subreddit: {filtered['subreddit'].value_counts().idxmax()}

            Provide:
            1. A direct answer
            2. One additional insight the user didn't ask for
            """

            with st.spinner("Analyzing..."):
                response = qa_model(prompt)[0]['generated_text']

            st.write("### 🤖 Answer")
            st.write(response)

    else:
        st.info("Not enough data for chatbot analysis.")

# -------------------------------
# Domain Comparison
# -------------------------------
if domain1 and domain2:
    st.subheader("⚔️ Activity Comparison")

    def get_ts(dom):
        d = compute_text_features(df[df['domain'].str.contains(dom, case=False, na=False)])
        return d.resample('D', on='created_utc').size()

    fig4, ax4 = plt.subplots()
    get_ts(domain1).plot(ax=ax4, label=domain1)
    get_ts(domain2).plot(ax=ax4, label=domain2)
    ax4.legend()
    st.pyplot(fig4)
