import streamlit as st
import requests
from PIL import Image
import io
import random

st.set_page_config(page_title="MoTune 🎵", page_icon="🎵", layout="centered")

# ⚠️ Replace this with your Render URL after deploying
BACKEND_URL = "https://moodetune-1.onrender.com"

MOOD_EMOJI = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "surprise": "😲", "fear": "😨", "disgust": "🤢", "neutral": "😐"
}
MOOD_COLOR = {
    "happy": "#FFD93D", "sad": "#5B8DEF", "angry": "#FF5C5C",
    "surprise": "#B983FF", "fear": "#7C6EFF", "disgust": "#7CB342", "neutral": "#B0BEC5"
}

def youtube_link(title, artist):
    query = f"{title} {artist}".replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}"

# Session state
if "mood" not in st.session_state:
    st.session_state.mood = None
    st.session_state.confidence = None
    st.session_state.songs = None
    st.session_state.camera_key = 0

def reassess():
    st.session_state.mood = None
    st.session_state.confidence = None
    st.session_state.songs = None
    st.session_state.camera_key += 1

# ---------------- UI ----------------
st.markdown("""
<style>
    /* Hide Streamlit's top toolbar/header, including the star and menu */
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    .stApp {
        background: linear-gradient(145deg, #08052a 0%, #160a52 45%, #32107a 100%);
        color: white;
    }

    .block-container {
        max-width: 760px;
        padding: 2rem 1.2rem 3rem 1.2rem;
    }

    .brand {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }

    .brand-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #d99cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        color: #d8ccff;
        font-size: 1.05rem;
        margin-top: .35rem;
    }

    .glass-card {
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 24px;
        padding: 1.4rem;
        box-shadow: 0 12px 35px rgba(0,0,0,.25);
        backdrop-filter: blur(12px);
        margin-bottom: 1.2rem;
    }

    .mood-card {
        text-align: center;
        border-radius: 28px;
        padding: 1.7rem;
        margin-bottom: 1.4rem;
        background: linear-gradient(135deg, rgba(181,91,255,.45), rgba(77,49,190,.55));
        border: 1px solid rgba(255,255,255,.20);
        box-shadow: 0 15px 40px rgba(0,0,0,.28);
    }

    .mood-emoji {
        font-size: 4rem;
        line-height: 1;
    }

    .mood-title {
        font-size: 1.8rem;
        font-weight: 750;
        margin-top: .7rem;
    }

    .confidence {
        color: #ddd4ff;
        font-size: .95rem;
        margin-top: .25rem;
    }

    .playlist-title {
        font-size: 1.55rem;
        font-weight: 750;
        margin: .5rem 0 1rem 0;
    }

    .song-card {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 13px 15px;
        margin: 10px 0;
        border-radius: 18px;
        background: rgba(255,255,255,.11);
        border: 1px solid rgba(255,255,255,.12);
    }

    .song-number {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #c56cff, #7340e8);
        font-weight: 700;
        flex-shrink: 0;
    }

    .song-info {
        flex: 1;
    }

    .song-name {
        font-weight: 700;
        font-size: 1.02rem;
    }

    .artist-name {
        color: #cfc4ef;
        font-size: .88rem;
        margin-top: 2px;
    }

    .stButton > button {
        border-radius: 16px !important;
        min-height: 50px !important;
        font-weight: 650 !important;
        font-size: 1rem !important;
        border: 1px solid rgba(255,255,255,.20) !important;
        background: linear-gradient(135deg, #9b4dff, #5d35d4) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(75,35,160,.30) !important;
    }

    .stButton > button:hover {
        border-color: rgba(255,255,255,.45) !important;
        transform: translateY(-1px);
    }

    div[data-testid="stCameraInput"] {
        background: rgba(255,255,255,.08);
        border-radius: 22px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,.15);
    }

    .footer {
        text-align: center;
        color: #aaa0c9;
        font-size: .82rem;
        margin-top: 2rem;
    }

    a {
        color: #e0b5ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">
    <div class="brand-title">🎵 MoodTune</div>
    <div class="brand-subtitle">Your Mood. Your Music.</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.mood is None:
    st.markdown("""
    <div class="glass-card">
        <h2 style="margin:0;">📸 Discover your mood</h2>
        <p style="color:#cfc4ef;margin:.4rem 0 0 0;">
            Take a photo and MoodTune will choose music that matches how you feel.
        </p>
    </div>
    """, unsafe_allow_html=True)
    photo = st.camera_input("Take a photo to detect your mood", key=f"cam_{st.session_state.camera_key}")

    if photo is not None:
        with st.spinner("Analysing your mood with CNN..."):
            try:
                img_bytes = photo.getvalue()
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
                    timeout=30
                )
                result = response.json()

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.mood = result["mood"]
                    st.session_state.confidence = result["confidence"]
                    st.session_state.songs = result["songs"]
                    st.rerun()

            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
                st.info("Make sure your Render backend is running!")

else:
    mood = st.session_state.mood
    color = MOOD_COLOR.get(mood, "#B0BEC5")
    emoji = MOOD_EMOJI.get(mood, "🙂")

    st.markdown(
        f"""
        <div class="mood-card">
            <div class="mood-emoji">{emoji}</div>
            <div class="mood-title">You're feeling {mood.capitalize()}</div>
            <div class="confidence">
                Mood confidence: {st.session_state.confidence:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="playlist-title">🎧 Music picked for your mood</div>', unsafe_allow_html=True)

    for i, song in enumerate(st.session_state.songs, 1):
        link = youtube_link(song["title"], song["artist"])
        st.markdown(
            f"""
            <div class="song-card">
                <div class="song-number">{i}</div>
                <div class="song-info">
                    <div class="song-name">{song["title"]}</div>
                    <div class="artist-name">{song["artist"]}</div>
                </div>
                <a href="{link}" target="_blank" style="text-decoration:none;font-size:1.5rem;">▶</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reassess Mood", use_container_width=True):
            reassess()
            st.rerun()
    with col2:
        if st.button("🔀 Shuffle Songs", use_container_width=True):
            random.shuffle(st.session_state.songs)
            st.rerun()

st.markdown("""
<div class="footer">
    MoodTune uses a CNN trained on the FER2013 dataset — 7 emotions, 35,000+ images.
</div>
""", unsafe_allow_html=True)
