"""
MoTune — Mood-based music recommender
Detects the user's facial emotion via webcam using DeepFace (CNN-based
emotion classifier) and recommends songs that match the detected mood.
Includes a "Reassess Mood" button so the user can re-scan at any time.
"""

import random

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="MoTune", page_icon="🎵", layout="centered")

st.markdown(
    """
    <style>
    .main { padding-top: 1.5rem; }
    .mood-card {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .mood-card h2 { margin: 0; font-size: 2rem; }
    .song-row {
        padding: 0.7rem 1rem;
        border-radius: 10px;
        background: #f4f4f8;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Song database — mapped to DeepFace's 7 emotion classes
# ----------------------------------------------------------------------
SONGS = {
    "happy": [
        ("Happy", "Pharrell Williams"),
        ("Can't Stop the Feeling!", "Justin Timberlake"),
        ("Good as Hell", "Lizzo"),
        ("Walking on Sunshine", "Katrina & The Waves"),
        ("Uptown Funk", "Mark Ronson ft. Bruno Mars"),
        ("Best Day of My Life", "American Authors"),
    ],
    "sad": [
        ("Someone Like You", "Adele"),
        ("Fix You", "Coldplay"),
        ("Skinny Love", "Bon Iver"),
        ("The Night We Met", "Lord Huron"),
        ("Liability", "Lorde"),
        ("Say Something", "A Great Big World"),
    ],
    "angry": [
        ("Break Stuff", "Limp Bizkit"),
        ("Killing in the Name", "Rage Against the Machine"),
        ("Duality", "Slipknot"),
        ("Bulls on Parade", "Rage Against the Machine"),
        ("Given Up", "Linkin Park"),
        ("Bodies", "Drowning Pool"),
    ],
    "surprise": [
        ("Bohemian Rhapsody", "Queen"),
        ("Feel It Still", "Portugal. The Man"),
        ("Levitating", "Dua Lipa"),
        ("Blinding Lights", "The Weeknd"),
        ("Electric Feel", "MGMT"),
        ("Sunflower", "Post Malone & Swae Lee"),
    ],
    "fear": [
        ("Breathe Me", "Sia"),
        ("Weightless", "Marconi Union"),
        ("Holocene", "Bon Iver"),
        ("Vienna", "Billy Joel"),
        ("River Flows in You", "Yiruma"),
        ("Clair de Lune", "Debussy"),
    ],
    "disgust": [
        ("Toxic", "Britney Spears"),
        ("You Oughta Know", "Alanis Morissette"),
        ("Irreplaceable", "Beyoncé"),
        ("Rolling in the Deep", "Adele"),
        ("Before He Cheats", "Carrie Underwood"),
        ("Bad Blood", "Taylor Swift"),
    ],
    "neutral": [
        ("Sunday Morning", "Maroon 5"),
        ("Banana Pancakes", "Jack Johnson"),
        ("Circles", "Post Malone"),
        ("Better Together", "Jack Johnson"),
        ("Golden", "Harry Styles"),
        ("Lost in Japan", "Shawn Mendes"),
    ],
}

EMOJI = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢",
    "neutral": "😐",
}


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def detect_mood(pil_image):
    """Run DeepFace's CNN emotion classifier on a PIL image.
    Returns (dominant_emotion, emotion_score_dict)."""
    img_array = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    result = DeepFace.analyze(
        img_bgr,
        actions=["emotion"],
        enforce_detection=False,  # don't crash if face is at an odd angle
        detector_backend="opencv",
    )
    if isinstance(result, list):
        result = result[0]

    return result["dominant_emotion"], result["emotion"]


def recommend_songs(mood, n=4):
    pool = SONGS.get(mood, SONGS["neutral"])
    return random.sample(pool, min(n, len(pool)))


def youtube_search_link(title, artist):
    query = f"{title} {artist}".replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}"


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key in ("mood", "scores", "songs"):
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🎵 MoTune")
st.caption("Music that matches your mood — read live from your face with a CNN emotion model.")

with st.sidebar:
    st.header("How it works")
    st.write(
        "1. MoTune takes a snapshot from your camera.\n"
        "2. A CNN (via the DeepFace library) classifies your facial "
        "expression into one of 7 emotions.\n"
        "3. MoTune matches that mood to a curated song list.\n"
        "4. Mood changed? Hit **Reassess Mood** to scan again."
    )
    st.divider()
    st.caption("First run downloads the emotion-detection model weights — needs an internet connection once.")

img_file = st.camera_input("Look at the camera to scan your mood")

if img_file is not None and st.session_state.mood is None:
    with st.spinner("Reading your mood..."):
        image = Image.open(img_file)
        try:
            mood, scores = detect_mood(image)
            st.session_state.mood = mood
            st.session_state.scores = scores
            st.session_state.songs = recommend_songs(mood)
        except Exception as e:
            st.error(f"Couldn't get a clear read on your face — try again with better lighting. ({e})")

if st.session_state.mood:
    mood = st.session_state.mood
    st.markdown(
        f"""<div class="mood-card"><h2>{EMOJI.get(mood, '')} {mood.capitalize()}</h2>
        <p>That's the mood MoTune is picking up right now.</p></div>""",
        unsafe_allow_html=True,
    )

    with st.expander("See confidence scores for every emotion"):
        st.bar_chart(st.session_state.scores)

    st.subheader("Recommended for you")
    for title, artist in st.session_state.songs:
        link = youtube_search_link(title, artist)
        st.markdown(
            f'<div class="song-row">🎧 <b>{title}</b> — {artist} &nbsp; '
            f'[<a href="{link}" target="_blank">Play</a>]</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔀 Shuffle songs for this mood"):
            st.session_state.songs = recommend_songs(mood)
            st.rerun()
    with col2:
        if st.button("🔄 Reassess Mood"):
            st.session_state.mood = None
            st.session_state.scores = None
            st.session_state.songs = None
            st.rerun()
else:
    st.info("Take a photo above to get your first mood-matched playlist.")
