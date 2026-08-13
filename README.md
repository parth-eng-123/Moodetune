# MoTune 🎵 — Mood-based Music Recommender

Detects your facial expression using a pretrained CNN (via the **DeepFace**
library) and recommends songs that fit your mood. Includes a **Reassess
Mood** button so you can re-check if your expression suddenly changes.

- **Frontend + logic:** Streamlit (`app.py`) — camera capture, UI, mood → song
  mapping, reassess/shuffle controls.
- **Mood detection:** DeepFace's pretrained emotion CNN (7 classes: happy,
  sad, angry, surprise, fear, disgust, neutral).
- **No API keys needed** — songs link out to YouTube search.

---

## Option A — Run it with no laptop at all (recommended for you)

Streamlit's camera widget works through your **phone's browser**, and you
can deploy the app for free without installing anything on your phone:

1. Create a free GitHub account (github.com) — can be done entirely from
   your phone browser.
2. Create a new repository and upload these 3 files: `app.py`,
   `requirements.txt`, and this `README.md`. GitHub's website lets you
   upload files directly, no computer needed.
3. Go to **share.streamlit.io**, sign in with GitHub, click **New app**,
   pick your repo, and set the main file to `app.py`. Click **Deploy**.
4. Wait 2–3 minutes for the first build (it installs DeepFace + TensorFlow).
5. Open the generated `https://your-app.streamlit.app` link on your phone,
   allow camera access when prompted, and MoTune is live.

Any device with a browser (phone, tablet, library computer, friend's
laptop) can now use it via that link.

## Option B — Run locally (if you get access to a computer)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL it prints (usually `http://localhost:8501`) — your
browser will ask for camera permission.

> First run downloads DeepFace's pretrained CNN weights (~a few MB), so it
> needs internet the first time. After that it's cached locally.

---

## How it works

1. `st.camera_input()` captures a photo from the browser's camera (works on
   desktop and mobile).
2. The photo is passed to `DeepFace.analyze(..., actions=["emotion"])`,
   which runs a pretrained convolutional neural network to score 7
   emotions and pick the dominant one.
3. `MOOD_SONGS` maps the dominant emotion to a curated list of songs; one
   is shown with a direct YouTube search link.
4. **Reassess Mood** clears the session state and shows a fresh camera
   widget, so if your expression changes you can re-scan and get new
   recommendations. **Shuffle Songs** keeps the same detected mood but
   picks a different set from that mood's song list.

## Customizing

- Edit `MOOD_SONGS` in `app.py` to swap in your own playlists (or wire up
  the Spotify Web API there instead of YouTube links if you get API keys).
- Swap `detector_backend="opencv"` for `"retinaface"` or `"mtcnn"` in
  `analyze_mood()` for more accurate face detection at the cost of speed.
- To train your own emotion CNN instead of using DeepFace's pretrained
  one (e.g. on the FER-2013 dataset), swap out `analyze_mood()` for a
  call to your own trained Keras model — ask and I can build that version
  too.
