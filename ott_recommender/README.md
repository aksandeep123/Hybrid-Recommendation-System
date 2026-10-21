# StreamPulse: Intelligent OTT Recommendation System Using Hybrid Filtering

A modern, high-performance Over-The-Top (OTT) media recommendation engine combining **Content-Based Filtering (TF-IDF + Cosine Similarity)** and **Collaborative Filtering (Matrix Factorization via Truncated SVD)** into a unified hybrid prediction framework.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 📌 Key Features
- **Dual Pipeline Hybrid Engine**: Integrates metadata-driven content similarity with latent behavioral matrix decomposition.
- **Dynamic Weight Tuning ($\alpha$)**: Real-time slider adjusting Content affinity vs Collaborative weighting.
- **Smart Catalog Filters**: Filter by genre, set minimum rating thresholds, and customize ranking metrics.
- **Dark OTT Aesthetic**: Polished UI styled after premier streaming platforms (Netflix, Disney+, Prime Video).
