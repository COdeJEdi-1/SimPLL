# Link Spread Intelligence Dashboard

This project is my assignment submission for the SimPPL internship.  
It is an investigative dashboard that analyses how **news domains spread across Reddit**, with a focus on:
- which communities amplify a link
- how discussion volume evolves over time
- what *tone* (sentiment, emotion, toxicity) those posts carry
- what kinds of topics/narratives they cluster into

The dashboard is built with **Streamlit** and hosted publicly.

---

## 1. Dataset

The input is a JSONL export of Reddit posts, preprocessed into `processed.csv`.  
Each row roughly corresponds to one Reddit post, with columns such as:

- `domain` – external link domain (e.g., `cnn.com`, `nypost.com`)
- `subreddit` – community where it was posted (e.g., `politics`, `Conservative`)
- `created_utc` – timestamp of the post
- `title` – post title / headline text

This assignment focuses on **link-level behaviour** rather than full comments.

---

## 2. What the Dashboard Shows

### 2.1 Summary Insights

After you type a domain (e.g. `cnn.com` or `nypost.com`) the app shows:

- **Total posts** mentioning that domain
- **Average sentiment** (VADER compound score on titles)
- **Average toxicity** (from a transformer toxicity classifier)
- Optional: compare with a **second domain** on the same screen

### 2.2 Time Series Trends

- A **line chart of posts per day** for the selected domain  
- When two domains are entered, a combined **“Posting Activity Comparison”** chart shows how attention rises and falls over time for both sources.

This directly addresses the rubric’s requirement for **time series of number of posts** and trends.

### 2.3 Communities / Subreddits

- A **Top Subreddits** bar chart summarises *which communities* amplify the chosen domain the most.
- This acts as a proxy for “communities or accounts that are key contributors” in the rubric.

### 2.4 Emotion & Tone

For every title, the app computes:

- **Sentiment** using `vaderSentiment`  
- **Toxicity** using `unitary/toxic-bert` (transformer classifier)  
- **Emotion** using `cardiffnlp/twitter-roberta-base-emotion`

These are aggregated into:

- An **Emotion Distribution** chart, and
- Summary cards showing **average sentiment** and **average toxicity**.

This satisfies the “Apply AI/ML” requirement, using multiple transformer-based models.

### 2.5 Topic Clustering (Narratives)

The dashboard runs a simple topic modelling / clustering step:

- Compute **TF–IDF** vectors over the titles
- Run **KMeans** to cluster them into a small number of groups
- Display a preview table (`title`, `cluster`) so the analyst can
  manually inspect the main themes.

This acts as a lightweight topic model and semantic grouping of narratives.

### 2.6 AI Narrative Summary

For each selected domain, the dashboard also generates an **“AI Summary Report”** that explains, in plain language:

- how frequently the domain appears
- whether the overall tone is positive / negative / mixed
- which subreddit is most responsible for amplification
- when activity peaked
- how many topic clusters were detected

This is meant for **non-technical audiences** and aligns with the rubric’s “GenAI summaries of the time-series plots” idea (implemented here as programmatic narrative rather than an external LLM API).

---

## 3. How to Use the Dashboard

1. Open the hosted app in the browser.
2. In **“Enter a news domain”**, type a domain like:
   - `cnn.com`
   - `nypost.com`
   - `youtube.com`
3. Optionally enter a **second domain** to compare.
4. Scroll down to explore:
   - Summary cards
   - Trend Over Time
   - Emotion Distribution
   - Top Subreddits
   - Topic Clustering table
   - AI Summary Report
   - Activity comparison for two domains

You can repeatedly change the domains to run **interactive “what-if” investigations**.

---

## 4. Technical Stack

- **Frontend / Dashboard:** Streamlit
- **Data processing:** pandas
- **Plots:** matplotlib + seaborn
- **ML / NLP models:**
  - `vaderSentiment` – sentiment analysis on titles
  - `unitary/toxic-bert` – toxicity classification
  - `cardiffnlp/twitter-roberta-base-emotion` – emotion classification
  - `sklearn` TF–IDF + KMeans – topic clustering

---

## 5. Running Locally

```bash
# 1. Clone repo
git clone <YOUR_REPO_URL>
cd <REPO_NAME>

# 2. Create env (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
