import streamlit as st
import requests
from PIL import Image
import io
import random

st.set_page_config(page_title="MoTune 🎵", page_icon="🎵", layout="centered")

# ⚠️ Replace this with your Render URL after deploying
BACKEND_URL = "https://moodtune-1.onrender.com"

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

# UI
st.title("🎵 MoTune")
st.caption("Mood-based music — powered by your trained CNN model")
st.divider()

if st.session_state.mood is None:
    st.subheader("📸 Look at the camera!")
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
        <div style="background:{color}22;border:2px solid {color};
        border-radius:16px;padding:24px;text-align:center;margin-bottom:16px;">
            <div style="font-size:52px;">{emoji}</div>
            <div style="font-size:24px;font-weight:700;color:{color};">
                You seem {mood.capitalize()}
            </div>
            <div style="font-size:14px;color:gray;">
                Confidence: {st.session_state.confidence:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🎧 Songs picked for you")
    for song in st.session_state.songs:
        link = youtube_link(song["title"], song["artist"])
        st.markdown(f"**{song['title']}** — {song['artist']}  \n[▶ Play on YouTube]({link})")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reassess Mood", use_container_width=True):
            reassess()
            st.rerun()
    with col2:
        if st.button("🔀 Shuffle Songs", use_container_width=True):
            st.rerun()

st.divider()
st.caption("MoTune uses a CNN trained on FER2013 dataset — 7 emotions, 35,000+ images")
