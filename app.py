import os
import ssl
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Hybrid Recommendation Systems Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Global Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stRadio > label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADERS & CACHING
# =========================================================
@st.cache_resource(show_spinner="🎬 Loading OTT streaming catalog & collaborative SVD models...")
def load_ott_engine():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        os.makedirs(data_dir, exist_ok=True)
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            zf.extract("ml-latest-small/movies.csv", data_dir)
            zf.extract("ml-latest-small/ratings.csv", data_dir)
            os.replace(os.path.join(data_dir, "ml-latest-small", "movies.csv"), movies_path)
            os.replace(os.path.join(data_dir, "ml-latest-small", "ratings.csv"), ratings_path)

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    movies = movies.merge(stats, on='movieId', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0.0).round(2)
    movies['rating_count'] = movies['rating_count'].fillna(0).astype(int)

    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False).fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])

    item_user = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=25, random_state=42)
    item_latent = svd.fit_transform(item_user)
    latent_df = pd.DataFrame(item_latent, index=item_user.index)

    all_genres = sorted(list({g for g_str in movies['genres'] if isinstance(g_str, str) for g in g_str.split('|') if g != '(no genres listed)'}))

    return movies, tfidf_matrix, latent_df, all_genres


@st.cache_resource(show_spinner="🎧 Loading Spotify tracks dataset & playlist SVD models...")
def load_music_engine():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(data_dir, "spotify_songs.csv")

    if not os.path.exists(csv_path):
        os.makedirs(data_dir, exist_ok=True)
        url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as r:
            with open(csv_path, 'wb') as f:
                f.write(r.read())

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['track_name', 'track_artist'])
    df_unique = df.drop_duplicates(subset=['track_name', 'track_artist']).copy()
    tracks_df = df_unique.sort_values(by='track_popularity', ascending=False).head(5000).reset_index(drop=True)

    audio_cols = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness']
    scaler = MinMaxScaler()
    audio_scaled = scaler.fit_transform(tracks_df[audio_cols])
    tempo_scaled = scaler.fit_transform(tracks_df[['tempo']])
    audio_matrix = np.hstack([audio_scaled, tempo_scaled])

    tracks_df['meta_corpus'] = (
        tracks_df['track_artist'] + " " +
        tracks_df['playlist_genre'].fillna('') + " " +
        tracks_df['playlist_subgenre'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english')
    meta_matrix = tfidf.fit_transform(tracks_df['meta_corpus'])

    df_top_occurrences = df[df['track_id'].isin(set(tracks_df['track_id']))]
    track_playlist_matrix = pd.crosstab(df_top_occurrences['track_id'], df_top_occurrences['playlist_id'])

    svd = TruncatedSVD(n_components=25, random_state=42)
    latent_item_matrix = svd.fit_transform(track_playlist_matrix)
    latent_df = pd.DataFrame(latent_item_matrix, index=track_playlist_matrix.index)

    genres_list = sorted([g for g in tracks_df['playlist_genre'].dropna().unique() if g])
    tracks_df['display_name'] = tracks_df['track_name'] + " — " + tracks_df['track_artist']

    return tracks_df, audio_matrix, meta_matrix, latent_df, genres_list


# =========================================================
# SIDEBAR APPLICATION SWITCHER
# =========================================================
st.sidebar.markdown("# 🚀 Recommendation Suite")
app_mode = st.sidebar.radio(
    "Choose Recommendation Engine:",
    ["🎬 StreamPulse (OTT Video)", "🎧 SoundWave (Music Audio)"],
    index=0
)
st.sidebar.markdown("---")


# =========================================================
# MODULE 1: OTT VIDEO RECOMMENDER (StreamPulse)
# =========================================================
if app_mode == "🎬 StreamPulse (OTT Video)":
    movies, tfidf_matrix, latent_df, all_genres = load_ott_engine()

    # Dynamic CSS for OTT
    st.markdown("""
    <style>
        .ott-brand {
            font-size: 2.3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #ff2a70 0%, #9b51e0 50%, #00d2ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }
        .ott-tagline { color: #8c9ba5; font-size: 0.95rem; margin-bottom: 22px; }
        .hero-card {
            background: linear-gradient(145deg, #1f1d36 0%, #14142b 100%);
            border: 1px solid #322f56;
            border-left: 6px solid #ff2a70;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 24px;
        }
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
        .rec-card {
            background: linear-gradient(145deg, #1a1a2e 0%, #161626 100%);
            border: 1px solid #2b2b48;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 16px;
            height: 100%;
        }
        .rec-card:hover {
            border-color: #ff2a70;
            box-shadow: 0 8px 24px rgba(255, 42, 112, 0.25);
        }
        .rec-rank { float: right; font-size: 1.2rem; font-weight: 800; color: #ff2a70; }
        .progress-box { background: #25283d; border-radius: 6px; height: 8px; width: 100%; overflow: hidden; margin: 6px 0; }
        .progress-fill { background: linear-gradient(90deg, #ff2a70, #9b51e0); height: 100%; border-radius: 6px; }
        .stat-card { background: #18192b; border: 1px solid #282a47; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 10px; }
        .stat-number { font-size: 1.4rem; font-weight: 800; color: #00d2ff; }
        .stat-label { font-size: 0.72rem; color: #8c9ba5; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### ⚙️ OTT Filter Settings")
    selected_genre = st.sidebar.selectbox("Filter by Genre", ["All"] + all_genres)
    num_recommendations = st.sidebar.slider("Number of Recommendations", 3, 20, 8)
    min_ratings = st.sidebar.slider("Min. Ratings Required", 0, 100, 15, step=5)
    sort_by = st.sidebar.selectbox("Sort Results By", ["Hybrid Score", "Rating Score", "Popularity"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ Hybrid Engine Balance")
    alpha = st.sidebar.slider(
        "Content (Genres) vs Collaborative (SVD) (α)",
        0.0, 1.0, 0.60, 0.05,
        help="α = 1.0 relies strictly on genre metadata. α = 0.0 relies on user interaction patterns."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Catalog Statistics")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(movies):,}</div><div class="stat-label">Titles</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(all_genres)}</div><div class="stat-label">Genres</div></div>', unsafe_allow_html=True)

    # Main Panel
    st.markdown('<div class="ott-brand">🎬 StreamPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="ott-tagline">AI-Powered OTT Recommendation Engine using Hybrid TF-IDF & Truncated SVD Matrix Factorization</div>', unsafe_allow_html=True)

    selected_title = st.selectbox("🔍 Search for an OTT Title / Movie:", movies['title'].values, index=0)
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        btn = st.button("🚀 Recommend Titles", use_container_width=True)

    if btn or selected_title:
        selected_idx = movies[movies['title'] == selected_title].index[0]
        target_movie = movies.iloc[selected_idx]
        target_id = target_movie['movieId']

        pills = "".join([f'<span class="genre-pill">{g}</span>' for g in target_movie['genres'].split('|')])
        st.markdown(f"""
        <div class="hero-card">
            <div style="font-size:1.3rem; font-weight:800; color:#fff; margin-bottom:4px;">📺 {target_movie['title']}</div>
            <div style="font-size:0.85rem; color:#f5a623; margin-bottom:10px;">★ {target_movie['avg_rating']:.2f} avg rating &nbsp;•&nbsp; 👥 {target_movie['rating_count']} reviews</div>
            <div>{pills}</div>
        </div>
        """, unsafe_allow_html=True)

        content_sim = cosine_similarity(tfidf_matrix[selected_idx], tfidf_matrix).flatten()

        if target_id in latent_df.index:
            vec = latent_df.loc[[target_id]].values
            collab_raw = cosine_similarity(vec, latent_df.values).flatten()
            collab_series = pd.Series(collab_raw, index=latent_df.index)
            collab_sim = movies['movieId'].map(collab_series).fillna(0.0).values
            collab_sim = np.clip((collab_sim + 1.0) / 2.0, 0.0, 1.0)
        else:
            collab_sim = np.zeros(len(movies))

        hybrid = (alpha * content_sim) + ((1.0 - alpha) * collab_sim)

        res_df = movies.copy()
        res_df['hybrid_score'] = hybrid
        res_df['content_score'] = content_sim
        res_df['collab_score'] = collab_sim
        res_df = res_df[res_df.index != selected_idx]

        if selected_genre != "All":
            res_df = res_df[res_df['genres'].str.contains(selected_genre, case=False, na=False)]
        if min_ratings > 0:
            res_df = res_df[res_df['rating_count'] >= min_ratings]

        if sort_by == "Hybrid Score":
            res_df = res_df.sort_values(by='hybrid_score', ascending=False)
        elif sort_by == "Rating Score":
            res_df = res_df.sort_values(by=['avg_rating', 'hybrid_score'], ascending=[False, False])
        elif sort_by == "Popularity":
            res_df = res_df.sort_values(by=['rating_count', 'hybrid_score'], ascending=[False, False])

        recs = res_df.head(num_recommendations)

        st.markdown(f"### 🎯 Top {len(recs)} Recommendations for '{selected_title}'")
        if recs.empty:
            st.warning("No titles match your filtering criteria. Try expanding the genre or lowering the ratings threshold.")
        else:
            rec_cols = st.columns(2)
            for i, (_, r) in enumerate(recs.iterrows(), start=1):
                with rec_cols[(i - 1) % 2]:
                    r_pills = "".join([f'<span class="genre-pill">{g}</span>' for g in r['genres'].split('|')])
                    pct = r['hybrid_score'] * 100.0
                    st.markdown(f"""
                    <div class="rec-card">
                        <span class="rec-rank">#{i}</span>
                        <div style="font-size:1.05rem; font-weight:700; color:#fff;">{r['title']}</div>
                        <div style="font-size:0.8rem; color:#a0aec0; margin-bottom:8px;">★ {r['avg_rating']:.1f} ({r['rating_count']} reviews)</div>
                        <div style="margin-bottom:8px;">{r_pills}</div>
                        <div style="font-size:0.75rem; color:#8c9ba5; display:flex; justify-content:space-between;">
                            <span>Similarity Match</span>
                            <span style="font-weight:700; color:#00d2ff;">{pct:.1f}%</span>
                        </div>
                        <div class="progress-box"><div class="progress-fill" style="width:{min(100.0, max(0.0, pct)):.1f}%;"></div></div>
                        <div style="font-size:0.72rem; color:#6c7d93;">Content: {r['content_score']*100:.1f}% &nbsp;|&nbsp; Collab: {r['collab_score']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================
# MODULE 2: MUSIC AUDIO RECOMMENDER (SoundWave)
# =========================================================
else:
    tracks_df, audio_matrix, meta_matrix, latent_df, genres_list = load_music_engine()

    # Dynamic CSS for Music
    st.markdown("""
    <style>
        .music-brand {
            font-size: 2.3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #1DB954 0%, #1ed760 50%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }
        .music-tagline { color: #b3b3b3; font-size: 0.95rem; margin-bottom: 22px; }
        .hero-music {
            background: linear-gradient(135deg, #18281d 0%, #181818 70%);
            border: 1px solid #282828;
            border-left: 6px solid #1DB954;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 24px;
        }
        .music-pill {
            display: inline-block;
            background: rgba(29, 185, 84, 0.15);
            color: #1ed760;
            border: 1px solid rgba(29, 185, 84, 0.35);
            padding: 3px 10px;
            border-radius: 16px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
            margin-top: 4px;
        }
        .music-card {
            background: #181818;
            border: 1px solid #282828;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 16px;
            height: 100%;
        }
        .music-card:hover {
            border-color: #1DB954;
            box-shadow: 0 8px 24px rgba(29, 185, 84, 0.2);
        }
        .music-rank { float: right; font-size: 1.2rem; font-weight: 800; color: #1DB954; }
        .m-progress-box { background: #2a2a2a; border-radius: 6px; height: 8px; width: 100%; overflow: hidden; margin: 6px 0; }
        .m-progress-fill { background: linear-gradient(90deg, #1DB954, #1ed760); height: 100%; border-radius: 6px; }
        .stat-card-m { background: #181818; border: 1px solid #282828; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 10px; }
        .stat-val-m { font-size: 1.4rem; font-weight: 800; color: #1DB954; }
        .stat-lbl-m { font-size: 0.72rem; color: #8c8c8c; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### 🎚️ Audio & Discovery Filters")
    selected_genre = st.sidebar.selectbox("Filter by Genre", ["All Genres"] + genres_list)
    tempo_filter = st.sidebar.selectbox("Tempo / BPM Pace", ["Any Pace", "Chill (<100 BPM)", "Upbeat (100-130 BPM)", "High Energy (>130 BPM)"])
    min_pop = st.sidebar.slider("Min. Spotify Popularity", 0, 100, 30, step=5)
    num_recommendations = st.sidebar.slider("Number of Recommendations", 3, 20, 8)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ Hybrid Recommendation Balance")
    alpha = st.sidebar.slider(
        "Acoustic Audio vs Playlist Co-occurrence (α)",
        0.0, 1.0, 0.60, 0.05,
        help="α = 1.0 relies purely on Acoustic DNA (energy, danceability, tempo). α = 0.0 relies on collective playlist groupings."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Music Library Stats")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card-m"><div class="stat-val-m">{len(tracks_df):,}</div><div class="stat-lbl-m">Tracks</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card-m"><div class="stat-val-m">{len(genres_list)}</div><div class="stat-lbl-m">Genres</div></div>', unsafe_allow_html=True)

    # Main Panel
    st.markdown('<div class="music-brand">🎧 SoundWave</div>', unsafe_allow_html=True)
    st.markdown('<div class="music-tagline">AI-Powered Music Recommendation Engine combining Acoustic DNA and Playlist Co-occurrence SVD</div>', unsafe_allow_html=True)

    default_idx = 0
    for cand in ["bad guy", "Circles", "The Box", "Shape of You", "Memories"]:
        matches = tracks_df[tracks_df['track_name'].str.contains(cand, case=False, na=False)].index
        if len(matches) > 0:
            default_idx = int(matches[0])
            break

    selected_track_name = st.selectbox("🎵 Search or Select Seed Track:", tracks_df['display_name'].values, index=default_idx)
    col_btn_m, _ = st.columns([1, 4])
    with col_btn_m:
        btn_m = st.button("✨ Discover Similar Music", use_container_width=True)

    if btn_m or selected_track_name:
        target_idx = tracks_df[tracks_df['display_name'] == selected_track_name].index[0]
        seed = tracks_df.iloc[target_idx]
        seed_id = seed['track_id']

        st.markdown(f"""
        <div class="hero-music">
            <div style="font-size:1.3rem; font-weight:800; color:#fff; margin-bottom:4px;">🎶 {seed['track_name']}</div>
            <div style="font-size:1rem; color:#1DB954; font-weight:600; margin-bottom:10px;">🎤 {seed['track_artist']}</div>
            <div style="margin-bottom:10px;">
                <span class="music-pill">Genre: {str(seed['playlist_genre']).upper()}</span>
                <span class="music-pill">Subgenre: {seed['playlist_subgenre']}</span>
                <span class="music-pill">🔥 {seed['track_popularity']} Pop</span>
                <span class="music-pill">⏱️ {seed['tempo']:.0f} BPM</span>
            </div>
            <div style="font-size:0.8rem; color:#b3b3b3; display:flex; gap:16px;">
                <span>🕺 Danceability: <b>{seed['danceability']*100:.0f}%</b></span>
                <span>⚡ Energy: <b>{seed['energy']*100:.0f}%</b></span>
                <span>🎭 Mood: <b>{seed['valence']*100:.0f}%</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        audio_sim = cosine_similarity(audio_matrix[target_idx:target_idx+1], audio_matrix).flatten()
        meta_sim = cosine_similarity(meta_matrix[target_idx], meta_matrix).flatten()
        content_scores = (0.60 * audio_sim) + (0.40 * meta_sim)

        if seed_id in latent_df.index:
            vec = latent_df.loc[[seed_id]].values
            collab_raw = cosine_similarity(vec, latent_df.values).flatten()
            collab_series = pd.Series(collab_raw, index=latent_df.index)
            collab_scores = tracks_df['track_id'].map(collab_series).fillna(0.0).values
            collab_scores = np.clip((collab_scores + 1.0) / 2.0, 0.0, 1.0)
        else:
            collab_scores = np.zeros(len(tracks_df))

        hybrid_m = (alpha * content_scores) + ((1.0 - alpha) * collab_scores)

        res_m = tracks_df.copy()
        res_m['hybrid_score'] = hybrid_m
        res_m['content_score'] = content_scores
        res_m['collab_score'] = collab_scores
        res_m = res_m[res_m.index != target_idx]

        if selected_genre != "All Genres":
            res_m = res_m[res_m['playlist_genre'].str.lower() == selected_genre.lower()]
        if min_pop > 0:
            res_m = res_m[res_m['track_popularity'] >= min_pop]
        if tempo_filter == "Chill (<100 BPM)":
            res_m = res_m[res_m['tempo'] < 100]
        elif tempo_filter == "Upbeat (100-130 BPM)":
            res_m = res_m[(res_m['tempo'] >= 100) & (res_m['tempo'] <= 130)]
        elif tempo_filter == "High Energy (>130 BPM)":
            res_m = res_m[res_m['tempo'] > 130]

        recs_m = res_m.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)

        st.markdown(f"### 🎯 Top {len(recs_m)} Recommendations for '{seed['track_name']}'")
        if recs_m.empty:
            st.warning("No tracks match your filtering criteria. Try expanding the tempo filter or lowering minimum popularity.")
        else:
            m_cols = st.columns(2)
            for i, (_, rm) in enumerate(recs_m.iterrows(), start=1):
                with m_cols[(i - 1) % 2]:
                    pct_m = rm['hybrid_score'] * 100.0
                    st.markdown(f"""
                    <div class="music-card">
                        <span class="music-rank">#{i}</span>
                        <div style="font-size:1.05rem; font-weight:700; color:#fff;">{rm['track_name']}</div>
                        <div style="font-size:0.85rem; color:#b3b3b3; margin-bottom:8px;">{rm['track_artist']}</div>
                        <div style="margin-bottom:8px;">
                            <span class="music-pill">{rm['playlist_genre'].upper()}</span>
                            <span class="music-pill">⚡ {rm['tempo']:.0f} BPM</span>
                            <span class="music-pill">🔥 {rm['track_popularity']} Pop</span>
                        </div>
                        <div style="font-size:0.75rem; color:#8c9ba5; display:flex; justify-content:space-between;">
                            <span>Acoustic Match</span>
                            <span style="font-weight:700; color:#1DB954;">{pct_m:.1f}%</span>
                        </div>
                        <div class="m-progress-box"><div class="m-progress-fill" style="width:{min(100.0, max(0.0, pct_m)):.1f}%;"></div></div>
                        <div style="font-size:0.72rem; color:#8c8c8c;">Audio: {rm['content_score']*100:.1f}% &nbsp;|&nbsp; Playlist: {rm['collab_score']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
