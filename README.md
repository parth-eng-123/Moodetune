# ðŸŽµ MoTune â€” Mood-Based Music Recommender

> Detects your facial emotion in real time using a CNN and recommends songs that match your mood.

---

## ðŸš€ Live Demo

ðŸŒ **Streamlit App:** [Open MoTune](https://moodetune-dxfr.streamlit.app)  
âš™ï¸ **Backend API:** [https://moodtune-1.onrender.com](https://moodtune-1.onrender.com)

---

## ðŸ§  How It Works

```
Your Face ðŸ“¸
     â†“
Streamlit (camera captures photo)
     â†“
FastAPI on Render (receives photo)
     â†“
ONNX CNN Model predicts emotion
     â†“
Returns mood + confidence + songs
     â†“
Streamlit displays result ðŸŽµ
```

---

## ðŸ—ï¸ Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Camera UI + Song display |
| Backend | FastAPI + Uvicorn | API server |
| ML Model | CNN (ONNX format) | Emotion detection |
| Hosting | Render.com | Backend deployment |
| Model Training | Google Colab | FER2013 dataset |

---

## ðŸ¤– CNN Model

- **Dataset:** FER2013 (35,000+ facial images)
- **Classes:** 7 emotions â€” angry, disgust, fear, happy, neutral, sad, surprise
- **Architecture:** 3x Conv2D + BatchNorm + MaxPooling + Dense layers
- **Format:** ONNX (lightweight, no TensorFlow needed at runtime)
- **Input:** 48x48 grayscale face image
- **Output:** Emotion probabilities (softmax)

---

## ðŸ“ Project Structure

```
Moodetune/
â”œâ”€â”€ app.py                  # Streamlit frontend
â”œâ”€â”€ main.py                 # FastAPI backend
â”œâ”€â”€ motune_model.onnx       # Trained CNN model (ONNX)
â”œâ”€â”€ motune_model.joblib     # Trained CNN model (joblib)
â”œâ”€â”€ requirements.txt        # Python dependencies
â”œâ”€â”€ render.yaml             # Render deployment config
â””â”€â”€ README.md               # This file
```

---

## âš™ï¸ Tech Stack

- **Python** â€” Core language
- **TensorFlow/Keras** â€” CNN model training
- **ONNX Runtime** â€” Lightweight model inference
- **OpenCV** â€” Image preprocessing
- **FastAPI** â€” REST API backend
- **Uvicorn** â€” ASGI server
- **Streamlit** â€” Frontend web app
- **Render** â€” Cloud deployment
- **Joblib** â€” Model serialization
- **Kaggle API** â€” Dataset download

---

## ðŸŽ¯ Features

- âœ… Real-time facial emotion detection via camera
- âœ… 7 emotion classes detection
- âœ… Song recommendations based on mood
- âœ… Reassess Mood button for re-detection
- âœ… Shuffle Songs button
- âœ… YouTube links for instant playback
- âœ… Mobile friendly
- âœ… Android APK available

---

## ðŸ› ï¸ Run Locally

```bash
# Clone repo
git clone https://github.com/parth-eng-123/Moodetune.git
cd Moodetune

# Install dependencies
pip install -r requirements.txt

# Run Streamlit frontend
streamlit run app.py

# Run FastAPI backend (separate terminal)
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

## ðŸ“Š Model Training

Model was trained in Google Colab on FER2013 dataset:

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(7, activation='softmax')
])
```

---

## ðŸ‘¨â€ðŸ’» Developer

**Parth** â€” GNIOT Capstone Project  
GitHub: [@parth-eng-123](https://github.com/parth-eng-123)

---

## ðŸ“„ License

This project is for educational purposes as part of the GNIOT Capstone Project.
