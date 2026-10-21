import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

base_dir = "e:/project/Hybrid-Recommendation-System"
img_dir = os.path.join(base_dir, "doc_images")
docx_path = os.path.join(base_dir, "SoundWave_Music_Recommendation_Report.docx")
pdf_path = os.path.join(base_dir, "SoundWave_Music_Recommendation_Report.pdf")

# =========================================================================
# 1. GENERATE WORD (.DOCX) DOCUMENT
# =========================================================================
def create_word_document():
    doc = Document()

    # Set Margins (1 inch all sides)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Header details
    p_header = doc.add_paragraph()
    r1 = p_header.add_run("NAME: - SANDEEP KUMAR")
    r1.bold = True
    r1.font.size = Pt(11)
    p_header.add_run("               ")
    r2 = p_header.add_run("DIV: - B/B")
    r2.bold = True
    r2.font.size = Pt(11)
    p_header.add_run("                     ")
    r3 = p_header.add_run("ID: - VU1F2324065")
    r3.bold = True
    r3.font.size = Pt(11)

    p_proj = doc.add_paragraph()
    r_proj = p_proj.add_run("MINI - PROJECT (BDA)")
    r_proj.bold = True
    r_proj.font.size = Pt(13)
    r_proj.font.color.rgb = RGBColor(29, 185, 84)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(6)
    r_t_lbl = p_title.add_run("TITLE: - ")
    r_t_lbl.bold = True
    r_t_lbl.font.size = Pt(11)
    r_t_val = p_title.add_run("SoundWave – Music Recommendation System Using Spotify 30k Dataset")
    r_t_val.font.size = Pt(11)

    # Aim
    p_aim = doc.add_paragraph()
    p_aim.paragraph_format.space_after = Pt(8)
    r_aim_lbl = p_aim.add_run("AIM: - ")
    r_aim_lbl.bold = True
    r_aim_lbl.font.size = Pt(11)
    r_aim_val = p_aim.add_run(
        "To design and develop a web-based Music Recommendation System using Python, Streamlit, and the Spotify 30k Audio Features dataset that recommends tracks based on user preferences using Hybrid Filtering (Content-Based Audio DNA Vectorization, TF-IDF, and Truncated SVD Playlist Collaborative Matrix Factorization). The project also aims to analyze a large music streaming dataset and visualize useful statistical and acoustic information through an interactive website."
    )
    r_aim_val.font.size = Pt(10.5)

    # Theory Section
    p_th = doc.add_paragraph()
    r_th = p_th.add_run("THEORY: -")
    r_th.bold = True
    r_th.font.size = Pt(11)

    # Theory Subsections
    theory_sections = [
        ("Introduction", 
         "A Recommendation System is a data analytics application that suggests relevant items to users based on their interests and preferences. Recommendation systems are widely used by platforms such as Spotify, Apple Music, YouTube Music, Netflix, and Amazon to help users discover content seamlessly.\n"
         "In this project, a Music Recommendation System named SoundWave is developed using Python and Streamlit. The system analyzes multi-dimensional acoustic properties (energy, danceability, valence, tempo) and user playlist co-occurrences to recommend tracks similar to those selected by the user."),

        ("Dataset Used",
         "The project uses the benchmark Spotify 30k Audio Features Dataset curated from the Spotify Web API.\n"
         "The dataset contains:\n"
         "• Total Tracks: 32,833 songs across diverse eras\n"
         "• Unique Playlists: 471 user-curated and editorial playlists\n"
         "• Core Music Genres: 6 major genres (Pop, Rap, Rock, Latin, R&B, EDM)\n"
         "• Audio Features: Continuous numerical attributes including Danceability, Energy, Valence (Musical Mood), Acousticness, Speechiness, Instrumentalness, and Tempo (BPM)\n"
         "The dataset contains track names, artists, popularity scores, subgenres, and acoustic signal characteristics. Pandas is used to load, clean, normalize, and analyze the data."),

        ("Content-Based Filtering",
         "Content-Based Filtering is a recommendation technique that suggests items based on similarities between their intrinsic features.\n"
         "In this project, both digital audio signal descriptors (danceability, energy, tempo, valence) and track metadata (genres, artists) serve as the primary features. When a user selects a seed song, the system normalizes its acoustic DNA using Min-Max Scaling and calculates mathematical proximity against all other songs in the catalog to discover acoustically matching tracks."),

        ("TF-IDF Vectorization",
         "TF-IDF stands for Term Frequency–Inverse Document Frequency.\n"
         "It is used to convert categorical metadata—including music genres, subgenres, and artist names—into high-dimensional numerical feature vectors.\n"
         "The technique assigns higher weights to rare or niche musical subgenres (such as 'progressive electro house' or 'neo soul') while diminishing the influence of ubiquitous terms. This ensures precise semantic alignment when pairing musical styles."),

        ("Cosine Similarity",
         "Cosine Similarity measures the similarity between two numerical vectors by calculating the cosine of the angle between them in a high-dimensional vector space.\n"
         "Its mathematical formula is:\n"
         "Cosine Similarity (A, B) = (A · B) / (||A|| × ||B||)\n"
         "A similarity score close to 1.0 indicates near-identical acoustic and genre characteristics. The system evaluates cosine similarity across the standardized audio matrices to retrieve songs that seamlessly match the rhythm and mood of the seed track."),

        ("Collaborative Filtering (Truncated SVD)",
         "To complement acoustic attributes with genuine listener behavior, the system factorizes the sparse Track-Playlist Co-occurrence Matrix using Truncated Singular Value Decomposition (SVD):\n"
         "R ≈ U · Σ · V^T\n"
         "This maps tracks into a 25-dimensional latent embedding space representing social listening habits and contextual playlist groupings, capturing cultural connections that audio signals alone cannot detect."),

        ("Hybrid Recommendation Engine",
         "The recommendation system dynamically synthesizes normalized scores from both the Content-Based Acoustic model and the Collaborative SVD model using a weighted linear fusion formula:\n"
         "Final Score = α · Score(Content) + (1 - α) · Score(Collaborative)\n"
         "By default, α = 0.60 (60% Acoustic DNA & Metadata, 40% Playlist Collaborative Affinity). This ensures newly released tracks without extensive playlist history can still be recommended accurately (mitigating the cold-start problem) while actively leveraging community playlist curation."),

        ("Website Implementation",
         "The system is developed using Streamlit, which provides an interactive, dark-mode web application inspired by Spotify's user interface.\n"
         "The website contains three main interactive modules:\n"
         "1. Music Recommendations & Audio DNA: Users search for or select any song, inspect its acoustic metrics (Danceability %, Energy %, Mood %, Tempo BPM), and receive the top 10 recommended tracks complete with progress match bars and feature breakdowns.\n"
         "2. Top Rated & Popular Tracks: Displays trending songs with dynamic filters for genre, minimum popularity score, and tempo pacing (Chill <100 BPM, Upbeat 100-130 BPM, High Energy >130 BPM).\n"
         "3. Dataset & Genre Analytics: Visualizes distribution charts including total tracks by genre, popularity distributions, and acoustic attribute correlations.")
    ]

    for title, content in theory_sections:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.space_before = Pt(6)
        p_t.paragraph_format.space_after = Pt(2)
        r_sub = p_t.add_run(title)
        r_sub.bold = True
        r_sub.font.size = Pt(11)

        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_after = Pt(6)
        r_txt = p_c.add_run(content)
        r_txt.font.size = Pt(10)

    # Technologies Used
    p_tech = doc.add_paragraph()
    r_tech = p_tech.add_run("TECHNOLOGIES USED")
    r_tech.bold = True
    r_tech.font.size = Pt(11)

    tech_items = [
        ("Python", "Main programming language for data analysis and ML algorithms"),
        ("Streamlit", "Web application development and responsive dark-mode UI"),
        ("Pandas", "Data loading, cleansing, and multi-dimensional matrix operations"),
        ("NumPy", "High-performance vector operations and linear algebra"),
        ("Scikit-learn", "MinMaxScaler, TfidfVectorizer, Cosine Similarity, and TruncatedSVD"),
        ("Matplotlib", "Data visualization and acoustic feature radar charts"),
        ("GitHub", "Project version control, source code hosting, and documentation")
    ]
    for k, v in tech_items:
        p_ti = doc.add_paragraph(style='List Bullet')
        p_ti.paragraph_format.space_after = Pt(2)
        r_k = p_ti.add_run(f"{k} – ")
        r_k.bold = True
        r_k.font.size = Pt(10)
        r_v = p_ti.add_run(v)
        r_v.font.size = Pt(10)

    # Advantages
    p_adv = doc.add_paragraph()
    p_adv.paragraph_format.space_before = Pt(6)
    r_adv_h = p_adv.add_run("Advantages")
    r_adv_h.bold = True
    r_adv_h.font.size = Pt(11)

    p_adv_c = doc.add_paragraph()
    p_adv_c.paragraph_format.space_after = Pt(6)
    r_adv_txt = p_adv_c.add_run(
        "The system overcomes the cold-start problem by utilizing multi-dimensional acoustic signals, maintains rhythmic and mood continuity via BPM and valence alignment, mitigates popularity bias through hybrid score normalization, and delivers sub-second inference speeds inside an intuitive Spotify-styled web portal."
    )
    r_adv_txt.font.size = Pt(10)

    # Limitations
    p_lim = doc.add_paragraph()
    r_lim_h = p_lim.add_run("Limitations")
    r_lim_h.bold = True
    r_lim_h.font.size = Pt(11)

    p_lim_c = doc.add_paragraph()
    p_lim_c.paragraph_format.space_after = Pt(6)
    r_lim_txt = p_lim_c.add_run(
        "Recommendations rely on high-level precomputed audio features rather than raw audio spectrograms or lyrics NLP analysis. Additionally, implicit skip behaviors during live listening sessions are not dynamically adapted in real-time."
    )
    r_lim_txt.font.size = Pt(10)

    # Conclusion
    p_con = doc.add_paragraph()
    r_con_h = p_con.add_run("CONCLUSION: - ")
    r_con_h.bold = True
    r_con_h.font.size = Pt(11)
    r_con_txt = p_con.add_run(
        "The SoundWave Music Recommendation System was successfully designed and implemented as a web-based Big Data Analytics mini project using the Spotify 30k dataset. The project demonstrates how high-dimensional acoustic feature engineering can be paired with Truncated SVD matrix factorization to generate personalized music recommendations. In the future, the system can be expanded using Deep Learning (Convolutional Neural Networks on audio spectrograms) and real-time session reinforcement learning."
    )
    r_con_txt.font.size = Pt(10)

    # Screenshots Section
    p_sc = doc.add_paragraph()
    p_sc.paragraph_format.space_before = Pt(10)
    r_sc = p_sc.add_run("SCREENSHOTS OF THE SYSTEM")
    r_sc.bold = True
    r_sc.font.size = Pt(11)

    screenshots = [
        ("Screenshot 1 – Music Recommendations & Acoustic DNA Comparison: -",
         os.path.join(img_dir, "figure1_radar.png"), "Figure 1: Radar plot of continuous acoustic attributes between seed track and top recommendation."),
        ("Screenshot 2 – Core Music Genre Distribution in Dataset: -",
         os.path.join(img_dir, "figure2_genres.png"), "Figure 2: Distribution of tracks across core musical taxonomies."),
        ("Screenshot 3 – Top 5 Recommendations Score Decomposition: -",
         os.path.join(img_dir, "figure3_scores.png"), "Figure 3: Hybrid score synthesis comparing Content (Acoustic) and Collaborative (Playlist SVD) affinities.")
    ]

    for title, img_p, cap in screenshots:
        p_st = doc.add_paragraph()
        p_st.paragraph_format.space_before = Pt(6)
        r_st = p_st.add_run(title)
        r_st.bold = True
        r_st.font.size = Pt(10.5)

        if os.path.exists(img_p):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_picture(img_p, width=Inches(5.0))
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_cap = p_cap.add_run(cap)
            r_cap.italic = True
            r_cap.font.size = Pt(9.5)
            r_cap.font.color.rgb = RGBColor(120, 120, 120)

    # Links
    links_data = [
        ("DEMO VIDEO: - ", "https://drive.google.com/file/d/sample_music_demo/view"),
        ("GITHUB (CODE BASE): - ", "https://github.com/aksandeep123/Hybrid-Recommendation-System"),
        ("HOSTED URL: - ", "https://hybrid-recommendation-system.streamlit.app")
    ]
    for lbl, val in links_data:
        p_l = doc.add_paragraph()
        p_l.paragraph_format.space_after = Pt(2)
        r_l = p_l.add_run(lbl)
        r_l.bold = True
        r_l.font.size = Pt(10)
        r_v = p_l.add_run(val)
        r_v.font.size = Pt(10)
        r_v.font.color.rgb = RGBColor(0, 102, 204)

    p_note = doc.add_paragraph()
    r_n = p_note.add_run("README also has in detail explanation of the topic.")
    r_n.italic = True
    r_n.font.size = Pt(9.5)

    doc.save(docx_path)
    print("Word document generated successfully at:", docx_path)


# =========================================================================
# 2. GENERATE PDF (.PDF) DOCUMENT
# =========================================================================
def create_pdf_document():
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_header = ParagraphStyle(
        'HeaderStyle',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#111111')
    )
    style_proj = ParagraphStyle(
        'ProjStyle',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1DB954'),
        spaceAfter=4
    )
    style_title = ParagraphStyle(
        'TitleStyle',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#111111'),
        spaceAfter=4
    )
    style_body = ParagraphStyle(
        'BodyCustom',
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=5
    )
    style_h2 = ParagraphStyle(
        'H2Custom',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#111111'),
        spaceBefore=5,
        spaceAfter=2
    )
    style_caption = ParagraphStyle(
        'CapCustom',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        alignment=1, # Center
        textColor=colors.HexColor('#666666'),
        spaceAfter=5
    )

    story = []

    # Header Row
    header_text = "<b>NAME: - SANDEEP KUMAR</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>DIV: - B/B</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>ID: - VU1F2324065</b>"
    story.append(Paragraph(header_text, style_header))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>MINI - PROJECT (BDA)</b>", style_proj))
    story.append(Spacer(1, 2))

    story.append(Paragraph("<b>TITLE: - </b>SoundWave – Music Recommendation System Using Spotify 30k Dataset", style_title))
    story.append(Paragraph(
        "<b>AIM: - </b>To design and develop a web-based Music Recommendation System using Python, Streamlit, and the Spotify 30k Audio Features dataset that recommends tracks based on user preferences using Hybrid Filtering (Content-Based Audio DNA Vectorization, TF-IDF, and Truncated SVD Playlist Collaborative Matrix Factorization). The project also aims to analyze a large music streaming dataset and visualize useful statistical and acoustic information through an interactive website.",
        style_body
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>THEORY: -</b>", style_title))

    sections = [
        ("Introduction", 
         "A Recommendation System is a data analytics application that suggests relevant items to users based on their interests and preferences. Recommendation systems are widely used by platforms such as Spotify, Apple Music, YouTube Music, Netflix, and Amazon to help users discover content seamlessly.<br/>"
         "In this project, a Music Recommendation System named SoundWave is developed using Python and Streamlit. The system analyzes multi-dimensional acoustic properties (energy, danceability, valence, tempo) and user playlist co-occurrences to recommend tracks similar to those selected by the user."),

        ("Dataset Used",
         "The project uses the benchmark Spotify 30k Audio Features Dataset curated from the Spotify Web API.<br/>"
         "The dataset contains:<br/>"
         "• Total Tracks: 32,833 songs across diverse eras<br/>"
         "• Unique Playlists: 471 user-curated and editorial playlists<br/>"
         "• Core Music Genres: 6 major genres (Pop, Rap, Rock, Latin, R&B, EDM)<br/>"
         "• Audio Features: Continuous numerical attributes including Danceability, Energy, Valence (Musical Mood), Acousticness, Speechiness, Instrumentalness, and Tempo (BPM)<br/>"
         "The dataset contains track names, artists, popularity scores, subgenres, and acoustic signal characteristics. Pandas is used to load, clean, normalize, and analyze the data."),

        ("Content-Based Filtering",
         "Content-Based Filtering is a recommendation technique that suggests items based on similarities between their intrinsic features.<br/>"
         "In this project, both digital audio signal descriptors (danceability, energy, tempo, valence) and track metadata (genres, artists) serve as the primary features. When a user selects a seed song, the system normalizes its acoustic DNA using Min-Max Scaling and calculates mathematical proximity against all other songs in the catalog to discover acoustically matching tracks."),

        ("TF-IDF Vectorization",
         "TF-IDF stands for Term Frequency–Inverse Document Frequency.<br/>"
         "It is used to convert categorical metadata—including music genres, subgenres, and artist names—into high-dimensional numerical feature vectors.<br/>"
         "The technique assigns higher weights to rare or niche musical subgenres (such as 'progressive electro house' or 'neo soul') while diminishing the influence of ubiquitous terms. This ensures precise semantic alignment when pairing musical styles."),

        ("Cosine Similarity",
         "Cosine Similarity measures the similarity between two numerical vectors by calculating the cosine of the angle between them in a high-dimensional vector space.<br/>"
         "Its mathematical formula is:<br/>"
         "<b>Cosine Similarity (A, B) = (A · B) / (||A|| × ||B||)</b><br/>"
         "A similarity score close to 1.0 indicates near-identical acoustic and genre characteristics. The system evaluates cosine similarity across the standardized audio matrices to retrieve songs that seamlessly match the rhythm and mood of the seed track."),

        ("Collaborative Filtering (Truncated SVD)",
         "To complement acoustic attributes with genuine listener behavior, the system factorizes the sparse Track-Playlist Co-occurrence Matrix using Truncated Singular Value Decomposition (SVD):<br/>"
         "<b>R ≈ U · Σ · V^T</b><br/>"
         "This maps tracks into a 25-dimensional latent embedding space representing social listening habits and contextual playlist groupings, capturing cultural connections that audio signals alone cannot detect."),

        ("Hybrid Recommendation Engine",
         "The recommendation system dynamically synthesizes normalized scores from both the Content-Based Acoustic model and the Collaborative SVD model using a weighted linear fusion formula:<br/>"
         "<b>Final Score = α · Score(Content) + (1 - α) · Score(Collaborative)</b><br/>"
         "By default, α = 0.60 (60% Acoustic DNA & Metadata, 40% Playlist Collaborative Affinity). This ensures newly released tracks without extensive playlist history can still be recommended accurately (mitigating the cold-start problem) while actively leveraging community playlist curation."),

        ("Website Implementation",
         "The system is developed using Streamlit, which provides an interactive, dark-mode web application inspired by Spotify's user interface.<br/>"
         "The website contains three main interactive modules:<br/>"
         "1. Music Recommendations & Audio DNA: Users search for or select any song, inspect its acoustic metrics, and receive top recommended tracks complete with progress match bars and feature breakdowns.<br/>"
         "2. Top Rated & Popular Tracks: Displays trending songs with dynamic filters for genre, minimum popularity score, and tempo pacing (Chill, Upbeat, High Energy).<br/>"
         "3. Dataset & Genre Analytics: Visualizes distribution charts including total tracks by genre, popularity distributions, and acoustic attribute correlations.")
    ]

    for h, txt in sections:
        story.append(Paragraph(f"<b>{h}</b>", style_h2))
        story.append(Paragraph(txt, style_body))

    story.append(Paragraph("<b>TECHNOLOGIES USED</b>", style_h2))
    tech_str = (
        "• <b>Python</b> – Main programming language for data analysis and ML algorithms<br/>"
        "• <b>Streamlit</b> – Web application development and responsive dark-mode UI<br/>"
        "• <b>Pandas</b> – Data loading, cleansing, and multi-dimensional matrix operations<br/>"
        "• <b>NumPy</b> – High-performance vector operations and linear algebra<br/>"
        "• <b>Scikit-learn</b> – MinMaxScaler, TfidfVectorizer, Cosine Similarity, and TruncatedSVD<br/>"
        "• <b>Matplotlib</b> – Data visualization and acoustic feature radar charts<br/>"
        "• <b>GitHub</b> – Project version control, source code hosting, and documentation"
    )
    story.append(Paragraph(tech_str, style_body))

    story.append(Paragraph("<b>Advantages</b>", style_h2))
    story.append(Paragraph("The system overcomes the cold-start problem by utilizing multi-dimensional acoustic signals, maintains rhythmic and mood continuity via BPM and valence alignment, mitigates popularity bias through hybrid score normalization, and delivers sub-second inference speeds inside an intuitive Spotify-styled web portal.", style_body))

    story.append(Paragraph("<b>Limitations</b>", style_h2))
    story.append(Paragraph("Recommendations rely on high-level precomputed audio features rather than raw audio spectrograms or lyrics NLP analysis. Additionally, implicit skip behaviors during live listening sessions are not dynamically adapted in real-time.", style_body))

    story.append(Paragraph("<b>CONCLUSION: - </b>The SoundWave Music Recommendation System was successfully designed and implemented as a web-based Big Data Analytics mini project using the Spotify 30k dataset. The project demonstrates how high-dimensional acoustic feature engineering can be paired with Truncated SVD matrix factorization to generate personalized music recommendations. In the future, the system can be expanded using Deep Learning (Convolutional Neural Networks on audio spectrograms) and real-time session reinforcement learning.", style_body))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>SCREENSHOTS OF THE SYSTEM</b>", style_title))

    figs = [
        ("Screenshot 1 – Music Recommendations & Acoustic DNA Comparison: -",
         os.path.join(img_dir, "figure1_radar.png"), "Figure 1: Radar plot of continuous acoustic attributes between seed track and top recommendation."),
        ("Screenshot 2 – Core Music Genre Distribution in Dataset: -",
         os.path.join(img_dir, "figure2_genres.png"), "Figure 2: Distribution of tracks across core musical taxonomies."),
        ("Screenshot 3 – Top 5 Recommendations Score Decomposition: -",
         os.path.join(img_dir, "figure3_scores.png"), "Figure 3: Hybrid score synthesis comparing Content (Acoustic) and Collaborative (Playlist SVD) affinities.")
    ]

    for title, img_p, cap in figs:
        story.append(Paragraph(f"<b>{title}</b>", style_h2))
        if os.path.exists(img_p):
            story.append(RLImage(img_p, width=290, height=140))
            story.append(Paragraph(cap, style_caption))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))
    links = (
        "<b>DEMO VIDEO: - </b><font color='#0066cc'>https://drive.google.com/file/d/sample_music_demo/view</font><br/>"
        "<b>GITHUB (CODE BASE): - </b><font color='#0066cc'>https://github.com/aksandeep123/Hybrid-Recommendation-System</font><br/>"
        "<i>README also has in detail explanation of the topic.</i><br/>"
        "<b>HOSTED URL: - </b><font color='#0066cc'>https://hybrid-recommendation-system.streamlit.app</font>"
    )
    story.append(Paragraph(links, style_body))

    doc.build(story)
    print("PDF document generated successfully at:", pdf_path)


if __name__ == "__main__":
    create_word_document()
    create_pdf_document()
