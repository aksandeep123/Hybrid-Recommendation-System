# SoundWave: AI Music Recommendation System Using Hybrid Filtering

A high-performance music recommendation engine inspired by modern audio streaming platforms (Spotify, Apple Music). Combines **Content-Based Acoustic DNA Vectorization (Energy, Danceability, Valence, Tempo, Artist/Genre TF-IDF)** with **Collaborative Playlist Co-occurrence Matrix Factorization (Truncated SVD)** into a unified real-time discovery engine.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app.py --server.port 8502
```

Navigate to `http://localhost:8502` to explore the Spotify-inspired dark UI!

---

## 📌 Key Architectural Highlights
- **Acoustic Audio DNA**: Normalizes danceability, energy, acousticness, valence (mood), and tempo across multi-dimensional feature space.
- **Collaborative Playlist Mining**: Models playlist-track co-occurrence networks via 25-factor Truncated SVD.
- **Parametric Hybrid Weighting ($\alpha$)**: Real-time slider to smoothly transition from pure acoustic similarity to pure social/playlist clustering.
- **Audio Filters**: Search by genre, filter by tempo/BPM categories (Chill, Upbeat, High Energy), and enforce minimum track popularity thresholds.
