import os
import urllib.request
import ssl
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------
# Page Configuration & Spotify Dark Aesthetic
# ---------------------------------------------------------
st.set_page_config(
    page_title="SoundWave | AI Music Recommendation Engine",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Spotify Dark Aesthetic (#121212, #1DB954)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700;900&family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background: #121212;
        color: #ffffff;
    }

    /* Brand Header */
    .brand-title {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }

    .brand-subtitle {
        color: #b3b3b3;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* Selected Track Hero Banner */
    .hero-track {
        background: linear-gradient(135deg, #18281d 0%, #181818 70%);
        border: 1px solid #282828;
        border-left: 6px solid #1DB954;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    .hero-track-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .hero-track-artist {
        font-size: 1.05rem;
        color: #1DB954;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Badges & Pills */
    .badge {
        display: inline-block;
        background: rgba(29, 185, 84, 0.15);
        color: #1ed760;
        border: 1px solid rgba(29, 185, 84, 0.35);
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
    }

    .badge-gray {
        display: inline-block;
        background: #282828;
        color: #e0e0e0;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
    }

    /* Track Recommendation Card */
    .track-card {
        background: #181818;
        border: 1px solid #282828;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        height: 100%;
    }

    .track-card:hover {
        background: #242424;
        border-color: #1DB954;
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.2);
    }

    .card-rank {
        float: right;
        font-size: 1.3rem;
        font-weight: 900;
        color: #1DB954;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .card-artist {
        font-size: 0.9rem;
        color: #b3b3b3;
        margin-bottom: 10px;
    }

    /* Audio Feature Gauge Container */
    .gauge-container {
        display: flex;
        justify-content: space-between;
        margin: 10px 0 6px 0;
        font-size: 0.78rem;
        color: #b3b3b3;
    }

    .gauge-bar-bg {
        background: #2a2a2a;
        height: 7px;
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 12px;
    }

    .gauge-bar-fill {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        height: 100%;
        border-radius: 4px;
    }

    /* Metric Card */
    .stat-card {
        background: #181818;
        border: 1px solid #282828;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        margin-bottom: 10px;
    }

    .stat-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: #1DB954;
    }

    .stat-lbl {
        font-size: 0.72rem;
        color: #8c8c8c;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Pipeline & Machine Learning Preparation
# ---------------------------------------------------------
@st.cache_resource(show_spinner="🎧 Initializing audio feature vector space & collaborative SVD models...")
def load_and_prepare_music_engine():
    """
    Loads Spotify tracks dataset, normalizes acoustic characteristics,
    and factorizes the playlist-track co-occurrence network using Truncated SVD.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(data_dir, "spotify_songs.csv")

    if not os.path.exists(csv_path):
        os.makedirs(data_dir, exist_ok=True)
        dataset_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(dataset_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as r:
            with open(csv_path, 'wb') as f:
                f.write(r.read())

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['track_name', 'track_artist'])

    # Deduplicate tracks by name and artist to prevent duplicate recommendations
    df_unique = df.drop_duplicates(subset=['track_name', 'track_artist']).copy()

    # Filter top 6,000 popular titles for real-time responsiveness
    tracks_df = df_unique.sort_values(by='track_popularity', ascending=False).head(6000).reset_index(drop=True)

    # 1. Normalize Multi-dimensional Acoustic Features
    audio_cols = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness']
    scaler = MinMaxScaler()
    audio_scaled = scaler.fit_transform(tracks_df[audio_cols])
    tempo_scaled = scaler.fit_transform(tracks_df[['tempo']])
    audio_matrix = np.hstack([audio_scaled, tempo_scaled])

    # 2. Text Vectorization for Metadata (Genre + Subgenre + Artist)
    tracks_df['meta_corpus'] = (
        tracks_df['track_artist'] + " " +
        tracks_df['playlist_genre'].fillna('') + " " +
        tracks_df['playlist_subgenre'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english')
    meta_matrix = tfidf.fit_transform(tracks_df['meta_corpus'])

    # 3. Collaborative Filtering via Playlist-Track Co-occurrence Matrix SVD
    df_top_occurrences = df[df['track_id'].isin(set(tracks_df['track_id']))]
    track_playlist_matrix = pd.crosstab(df_top_occurrences['track_id'], df_top_occurrences['playlist_id'])

    svd = TruncatedSVD(n_components=25, random_state=42)
    latent_item_matrix = svd.fit_transform(track_playlist_matrix)
    latent_item_df = pd.DataFrame(latent_item_matrix, index=track_playlist_matrix.index)

    # Distinct Genres
    genres_list = sorted([g for g in tracks_df['playlist_genre'].dropna().unique() if g])

    # Display label for selectbox
    tracks_df['display_name'] = tracks_df['track_name'] + " — " + tracks_df['track_artist']

    return tracks_df, audio_matrix, meta_matrix, latent_item_df, genres_list


tracks_df, audio_matrix, meta_matrix, latent_item_df, genres_list = load_and_prepare_music_engine()


# ---------------------------------------------------------
# Sidebar Controls & Audio Filters
# ---------------------------------------------------------
st.sidebar.markdown("### 🎚️ Audio & Discovery Filters")

selected_genre = st.sidebar.selectbox("Filter by Genre", ["All Genres"] + genres_list)
tempo_filter = st.sidebar.selectbox("Tempo / BPM Pace", ["Any Pace", "Chill (<100 BPM)", "Upbeat (100-130 BPM)", "High Energy (>130 BPM)"])
min_popularity = st.sidebar.slider("Min. Spotify Popularity", min_value=0, max_value=100, value=30, step=5)
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=20, value=8)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Hybrid Recommendation Balance")
alpha = st.sidebar.slider(
    "Acoustic / Content vs Collaborative Factor (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
    help="α = 1.0 relies strictly on Audio characteristics & Artist metadata. α = 0.0 relies purely on Playlist co-occurrence patterns. 0.5-0.7 yields optimal hybrid discovery."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Music Library Stats")
c1, c2 = st.sidebar.columns(2)
with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val">{len(tracks_df):,}</div>
        <div class="stat-lbl">Active Tracks</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val">{len(genres_list)}</div>
        <div class="stat-lbl">Core Genres</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Application Dashboard
# ---------------------------------------------------------
st.markdown('<div class="brand-title">🎧 SoundWave</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">AI-Powered Music Recommendation Engine combining Acoustic DNA and Playlist Collaborative Intelligence</div>', unsafe_allow_html=True)

# Track Selector
default_track_idx = 0
# Look for a recognizable track index
for candidate in ["bad guy", "Circles", "The Box", "Shape of You", "Memories"]:
    matches = tracks_df[tracks_df['track_name'].str.contains(candidate, case=False, na=False)].index
    if len(matches) > 0:
        default_track_idx = int(matches[0])
        break

selected_display = st.selectbox(
    "🎵 Search or Select Seed Track:",
    tracks_df['display_name'].values,
    index=default_track_idx
)

col_run, _ = st.columns([1, 4])
with col_run:
    recommend_button = st.button("✨ Discover Similar Tracks", use_container_width=True)

if recommend_button or selected_display:
    target_idx = tracks_df[tracks_df['display_name'] == selected_display].index[0]
    seed_track = tracks_df.iloc[target_idx]
    seed_track_id = seed_track['track_id']

    # ---------------------------------------------------------
    # Display Selected Track Audio Hero Card
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class="hero-track">
        <div class="hero-track-title">🎶 {seed_track['track_name']}</div>
        <div class="hero-track-artist">🎤 {seed_track['track_artist']}</div>
        <div style="margin-bottom: 12px;">
            <span class="badge">Genre: {str(seed_track['playlist_genre']).upper()}</span>
            <span class="badge-gray">Subgenre: {seed_track['playlist_subgenre']}</span>
            <span class="badge-gray">🔥 Popularity: {seed_track['track_popularity']}/100</span>
            <span class="badge-gray">⏱️ Tempo: {seed_track['tempo']:.0f} BPM</span>
        </div>
        <div style="font-size: 0.8rem; color: #b3b3b3; display: flex; gap: 20px;">
            <span>🕺 Danceability: <b>{seed_track['danceability']*100:.0f}%</b></span>
            <span>⚡ Energy: <b>{seed_track['energy']*100:.0f}%</b></span>
            <span>🎭 Mood (Valence): <b>{seed_track['valence']*100:.0f}%</b></span>
            <span>🎙️ Acousticness: <b>{seed_track['acousticness']*100:.0f}%</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Compute Hybrid Recommendation Scores
    # ---------------------------------------------------------
    # 1. Content: Audio Feature Similarity + Metadata Similarity
    audio_sim = cosine_similarity(audio_matrix[target_idx:target_idx+1], audio_matrix).flatten()
    meta_sim = cosine_similarity(meta_matrix[target_idx], meta_matrix).flatten()
    content_scores = (0.60 * audio_sim) + (0.40 * meta_sim)

    # 2. Collaborative: Playlist Co-occurrence SVD Cosine Similarity
    if seed_track_id in latent_item_df.index:
        seed_latent_vec = latent_item_df.loc[[seed_track_id]].values
        collab_raw = cosine_similarity(seed_latent_vec, latent_item_df.values).flatten()
        collab_series = pd.Series(collab_raw, index=latent_item_df.index)
        collab_scores = tracks_df['track_id'].map(collab_series).fillna(0.0).values
        # Normalize into [0, 1]
        collab_scores = np.clip((collab_scores + 1.0) / 2.0, 0.0, 1.0)
    else:
        # Fallback for isolated tracks with zero co-playlist occurrences
        collab_scores = np.zeros(len(tracks_df))

    # 3. Dynamic Hybrid Fusion Score
    hybrid_scores = (alpha * content_scores) + ((1.0 - alpha) * collab_scores)

    results_df = tracks_df.copy()
    results_df['hybrid_score'] = hybrid_scores
    results_df['content_score'] = content_scores
    results_df['collab_score'] = collab_scores

    # Exclude the seed track itself
    results_df = results_df[results_df.index != target_idx]

    # Apply Sidebar Filters
    if selected_genre != "All Genres":
        results_df = results_df[results_df['playlist_genre'].str.lower() == selected_genre.lower()]

    if min_popularity > 0:
        results_df = results_df[results_df['track_popularity'] >= min_popularity]

    if tempo_filter == "Chill (<100 BPM)":
        results_df = results_df[results_df['tempo'] < 100]
    elif tempo_filter == "Upbeat (100-130 BPM)":
        results_df = results_df[(results_df['tempo'] >= 100) & (results_df['tempo'] <= 130)]
    elif tempo_filter == "High Energy (>130 BPM)":
        results_df = results_df[results_df['tempo'] > 130]

    # Rank and retrieve top recommendations
    recommendations = results_df.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)

    st.markdown(f"### 🎯 Top {len(recommendations)} Tracks Matched with '{seed_track['track_name']}'")

    if recommendations.empty:
        st.warning("No tracks matched your specific filtering criteria. Try expanding the tempo filter or lowering minimum popularity.")
    else:
        # Display in a 2-column grid
        rec_cols = st.columns(2)
        for i, (_, rec) in enumerate(recommendations.iterrows(), start=1):
            with rec_cols[(i - 1) % 2]:
                match_pct = rec['hybrid_score'] * 100.0

                st.markdown(f"""
                <div class="track-card">
                    <span class="card-rank">#{i}</span>
                    <div class="card-title">{rec['track_name']}</div>
                    <div class="card-artist">{rec['track_artist']}</div>
                    <div style="margin-bottom: 8px;">
                        <span class="badge">{rec['playlist_genre'].upper()}</span>
                        <span class="badge-gray">{rec['playlist_subgenre']}</span>
                        <span class="badge-gray">⚡ {rec['tempo']:.0f} BPM</span>
                        <span class="badge-gray">🔥 {rec['track_popularity']} Pop</span>
                    </div>
                    <div class="gauge-container">
                        <span>Acoustic Match</span>
                        <span style="font-weight: 700; color: #1DB954;">{match_pct:.1f}%</span>
                    </div>
                    <div class="gauge-bar-bg">
                        <div class="gauge-bar-fill" style="width: {min(100.0, max(0.0, match_pct)):.1f}%;"></div>
                    </div>
                    <div style="font-size: 0.72rem; color: #8c8c8c; display: flex; justify-content: space-between;">
                        <span>Audio Match: {rec['content_score']*100:.1f}%</span>
                        <span>Playlist Affinity: {rec['collab_score']*100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
