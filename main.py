import numpy as np
import cv2
import random
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MoTune API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load ONNX model - no TensorFlow needed!
session = ort.InferenceSession("motune_model.onnx")
input_name = session.get_inputs()[0].name

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

SONGS = {
    "happy": [
        {"title": "Happy", "artist": "Pharrell Williams"},
        {"title": "Uptown Funk", "artist": "Bruno Mars"},
        {"title": "Can't Stop the Feeling", "artist": "Justin Timberlake"},
        {"title": "Good as Hell", "artist": "Lizzo"},
        {"title": "Walking on Sunshine", "artist": "Katrina & The Waves"},
    ],
    "sad": [
        {"title": "Someone Like You", "artist": "Adele"},
        {"title": "Fix You", "artist": "Coldplay"},
        {"title": "Let Her Go", "artist": "Passenger"},
        {"title": "The Night We Met", "artist": "Lord Huron"},
        {"title": "Hello", "artist": "Adele"},
    ],
    "angry": [
        {"title": "In The End", "artist": "Linkin Park"},
        {"title": "Lose Yourself", "artist": "Eminem"},
        {"title": "Till I Collapse", "artist": "Eminem"},
        {"title": "Numb", "artist": "Linkin Park"},
        {"title": "Stronger", "artist": "Kanye West"},
    ],
    "fear": [
        {"title": "Brave", "artist": "Sara Bareilles"},
        {"title": "Stronger", "artist": "Kelly Clarkson"},
        {"title": "Weightless", "artist": "Marconi Union"},
        {"title": "Fix You", "artist": "Coldplay"},
        {"title": "Fight Song", "artist": "Rachel Platten"},
    ],
    "disgust": [
        {"title": "Roar", "artist": "Katy Perry"},
        {"title": "Titanium", "artist": "David Guetta ft. Sia"},
        {"title": "Confident", "artist": "Demi Lovato"},
        {"title": "Fight Song", "artist": "Rachel Platten"},
        {"title": "Shake It Off", "artist": "Taylor Swift"},
    ],
    "surprise": [
        {"title": "Bohemian Rhapsody", "artist": "Queen"},
        {"title": "Levitating", "artist": "Dua Lipa"},
        {"title": "Blinding Lights", "artist": "The Weeknd"},
        {"title": "Electric Feel", "artist": "MGMT"},
        {"title": "Feel It Still", "artist": "Portugal. The Man"},
    ],
    "neutral": [
        {"title": "Sunflower", "artist": "Post Malone"},
        {"title": "Watermelon Sugar", "artist": "Harry Styles"},
        {"title": "Golden Hour", "artist": "JVKE"},
        {"title": "Stay", "artist": "The Kid LAROI"},
        {"title": "Circles", "artist": "Post Malone"},
    ],
}

@app.get("/")
def home():
    return {"message": "MoTune API is running! 🎵"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Could not read image"}

    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1).astype(np.float32)

    # Run ONNX inference
    pred = session.run(None, {input_name: img})[0]
    emotion = emotions[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)
    songs = random.sample(SONGS.get(emotion, SONGS["neutral"]), 3)

    return {
        "mood": emotion,
        "confidence": round(confidence, 2),
        "songs": songs
    }
