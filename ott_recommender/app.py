import os
import urllib.request
import ssl
import zipfile
import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------
# Page Configuration & Modern OTT Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="StreamPulse | OTT Content Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark OTT Aesthetic (Netflix / CineMatch style)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 10% 20%, #17182e 0%, #0d0f18 90%);
        color: #e0e6ed;
    }

    /* OTT Header Gradient */
    .ott-brand {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff2a70 0%, #9b51e0 50%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        margin-bottom: 0px;
    }

    .ott-tagline {
        color: #8c9ba5;
        font-size: 0.95rem;
        margin-bottom: 25px;
    }

    /* Selected Title Hero Card */
    .hero-card {
        background: linear-gradient(145deg, #1f1d36 0%, #14142b 100%);
        border: 1px solid #322f56;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    .hero-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .hero-meta {
        font-size: 0.85rem;
        color: #f5a623;
        margin-bottom: 12px;
    }

    /* Genre Pill Badge */
    .genre-pill {
        display: inline-block;
        background: rgba(155, 81, 224, 0.2);
        color: #c77dff;
        border: 1px solid rgba(155, 81, 224, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
    }

    /* Recommendation Card */
    .rec-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #161626 100%);
        border: 1px solid #2b2b48;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        height: 100%;
    }

    .rec-card:hover {
        transform: translateY(-3px);
        border-color: #ff2a70;
        box-shadow: 0 8px 25px rgba(255, 42, 112, 0.25);
    }

    .rec-rank {
        float: right;
        font-size: 1.2rem;
        font-weight: 800;
        color: rgba(255, 42, 112, 0.8);
    }

    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .rec-stats {
        font-size: 0.8rem;
        color: #a0aec0;
        margin-bottom: 8px;
    }

    /* Custom Progress Bar Container */
    .progress-box {
        background: #25283d;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 6px 0;
    }

    .progress-fill {
        background: linear-gradient(90deg, #ff2a70, #9b51e0);
        height: 100%;
        border-radius: 6px;
    }

    /* Metric Stat Card in Sidebar */
    .stat-card {
        background: #18192b;
        border: 1px solid #282a47;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 10px;
    }

    .stat-number {
        font-size: 1.5rem;
        font-weight: 800;
        color: #00d2ff;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #8c9ba5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Ingestion & Machine Learning Pipeline
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing OTT recommendation engine & ML matrices...")
def load_and_prepare_ott_data():
    """
    Downloads and caches MovieLens streaming catalog metadata.
    Constructs both Content (TF-IDF) and Collaborative (SVD) latent matrices.
    """
    local_data_dir = os.path.join(os.path.dirname(__file__), "data", "ml-latest-small")
    movies_path = os.path.join(local_data_dir, "movies.csv")
    ratings_path = os.path.join(local_data_dir, "ratings.csv")

    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        os.makedirs(local_data_dir, exist_ok=True)
        dataset_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(dataset_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            zf.extract("ml-latest-small/movies.csv", os.path.dirname(local_data_dir))
            zf.extract("ml-latest-small/ratings.csv", os.path.dirname(local_data_dir))

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # 1. Calculate Aggregate Streaming Interaction Stats
    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    movies = movies.merge(stats, on='movieId', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0.0).round(2)
    movies['rating_count'] = movies['rating_count'].fillna(0).astype(int)

    # Clean genres metadata for Content-Based TF-IDF Vectorization
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False).fillna('')

    # 2. Content-Based TF-IDF Model
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])

    # 3. Collaborative Filtering via Truncated SVD Matrix Factorization
    item_user_matrix = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=25, random_state=42)
    item_latent_matrix = svd.fit_transform(item_user_matrix)
    item_latent_df = pd.DataFrame(item_latent_matrix, index=item_user_matrix.index)

    # Extract all distinct genres
    all_genres = sorted(list({genre for g_str in movies['genres'] if isinstance(g_str, str) for genre in g_str.split('|') if genre != '(no genres listed)'}))

    return movies, tfidf_matrix, item_latent_df, all_genres


movies, tfidf_matrix, item_latent_df, all_genres = load_and_prepare_ott_data()


# ---------------------------------------------------------
# Sidebar Controls & Filters
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Settings")

selected_genre = st.sidebar.selectbox("Filter by Genre", ["All"] + all_genres)
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=20, value=8)
min_ratings = st.sidebar.slider("Min. Ratings Required", min_value=0, max_value=100, value=15, step=5)
sort_by = st.sidebar.selectbox("Sort Results By", ["Hybrid Score", "Rating Score", "Popularity (Rating Count)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Hybrid Engine Balance")
alpha = st.sidebar.slider(
    "Content Affinity vs Collaborative Weight (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="α = 1.0 is Pure Content-Based (genres). α = 0.0 is Pure Collaborative Filtering (user interaction patterns). 0.5-0.7 provides an optimal hybrid balance."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(movies):,}</div>
        <div class="stat-label">Total Titles</div>
    </div>
    """, unsafe_allow_html=True)
with col_s2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(all_genres)}</div>
        <div class="stat-label">Genres</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Main UI Application
# ---------------------------------------------------------
st.markdown('<div class="ott-brand">🎬 StreamPulse</div>', unsafe_allow_html=True)
st.markdown('<div class="ott-tagline">Next-Generation OTT Media Recommendation Engine Powered by Hybrid Content-Collaborative Intelligence</div>', unsafe_allow_html=True)

# Search & Selection
selected_title = st.selectbox(
    "🔍 Search for an OTT Title / Movie:",
    movies['title'].values,
    index=0
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    generate_btn = st.button("🚀 Recommend", use_container_width=True)

if generate_btn or selected_title:
    selected_idx = movies[movies['title'] == selected_title].index[0]
    target_movie = movies.iloc[selected_idx]
    target_movie_id = target_movie['movieId']

    # Display Selected Movie Hero Card
    genres_pills = "".join([f'<span class="genre-pill">{g}</span>' for g in target_movie['genres'].split('|')])
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-title">📺 {target_movie['title']}</div>
        <div class="hero-meta">★ {target_movie['avg_rating']:.2f} avg rating &nbsp;•&nbsp; 👥 {target_movie['rating_count']} user ratings</div>
        <div>{genres_pills}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Hybrid Recommendation Computation
    # ---------------------------------------------------------
    # 1. Content-Based Cosine Similarity
    content_sim_scores = cosine_similarity(tfidf_matrix[selected_idx], tfidf_matrix).flatten()

    # 2. Collaborative Item-Item SVD Cosine Similarity
    if target_movie_id in item_latent_df.index:
        target_vec = item_latent_df.loc[[target_movie_id]].values
        collab_raw = cosine_similarity(target_vec, item_latent_df.values).flatten()
        collab_series = pd.Series(collab_raw, index=item_latent_df.index)
        collab_sim_scores = movies['movieId'].map(collab_series).fillna(0.0).values
        # Normalize from [-1, 1] to [0, 1]
        collab_sim_scores = np.clip((collab_sim_scores + 1.0) / 2.0, 0.0, 1.0)
    else:
        # Cold start fallback for items with no rating vectors
        collab_sim_scores = np.zeros(len(movies))

    # 3. Hybrid Weighted Score Calculation
    hybrid_scores = (alpha * content_sim_scores) + ((1.0 - alpha) * collab_sim_scores)

    results_df = movies.copy()
    results_df['hybrid_score'] = hybrid_scores
    results_df['content_score'] = content_sim_scores
    results_df['collab_score'] = collab_sim_scores

    # Filter out target movie
    results_df = results_df[results_df.index != selected_idx]

    # Apply Sidebar Filters
    if selected_genre != "All":
        results_df = results_df[results_df['genres'].str.contains(selected_genre, case=False, na=False)]

    if min_ratings > 0:
        results_df = results_df[results_df['rating_count'] >= min_ratings]

    # Sorting logic
    if sort_by == "Hybrid Score":
        results_df = results_df.sort_values(by='hybrid_score', ascending=False)
    elif sort_by == "Rating Score":
        results_df = results_df.sort_values(by=['avg_rating', 'hybrid_score'], ascending=[False, False])
    elif sort_by == "Popularity (Rating Count)":
        results_df = results_df.sort_values(by=['rating_count', 'hybrid_score'], ascending=[False, False])

    recommendations = results_df.head(num_recommendations)

    st.markdown(f"### 🎯 Top {len(recommendations)} Recommendations for '{selected_title}'")

    if recommendations.empty:
        st.warning("No titles match your filtering criteria. Try reducing the 'Min. Ratings Required' slider or clearing the genre filter.")
    else:
        # Display in a 2-column grid
        rec_cols = st.columns(2)
        for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
            with rec_cols[(i - 1) % 2]:
                pill_html = "".join([f'<span class="genre-pill">{g}</span>' for g in row['genres'].split('|')])
                score_pct = row['hybrid_score'] * 100.0

                stars_full = int(round(row['avg_rating']))
                star_display = "★" * stars_full + "☆" * (5 - stars_full)

                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-rank">#{i}</span>
                    <div class="rec-title">{row['title']}</div>
                    <div class="rec-stats">★ {row['avg_rating']:.1f} ({row['rating_count']} ratings) &nbsp;<span style="color:#f5a623;">{star_display}</span></div>
                    <div style="margin-bottom: 8px;">{pill_html}</div>
                    <div style="font-size: 0.78rem; color: #8c9ba5; display: flex; justify-content: space-between;">
                        <span>Similarity Match</span>
                        <span style="font-weight: 700; color: #00d2ff;">{score_pct:.1f}%</span>
                    </div>
                    <div class="progress-box">
                        <div class="progress-fill" style="width: {min(100.0, max(0.0, score_pct)):.1f}%;"></div>
                    </div>
                    <div style="font-size: 0.72rem; color: #6c7d93; margin-top: 4px;">
                        Content: {row['content_score']*100:.1f}% &nbsp;|&nbsp; Collab: {row['collab_score']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
