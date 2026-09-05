# MoTune - Mood-Based Music Recommender

Detects your facial emotion in real time using a CNN and recommends songs that match your mood.

---

## Live Demo

- Streamlit App: https://moodetune-dxfr.streamlit.app
- Backend API: https://moodtune-1.onrender.com

---

## How It Works

```
Your Face (Camera)
      |
Streamlit (captures photo)
      |
FastAPI on Render (receives photo)
      |
ONNX CNN Model (predicts emotion)
      |
Returns mood + confidence + songs
      |
Streamlit displays result
```

---

## Architecture

| Component      | Technology         | Purpose                  |
|----------------|--------------------|--------------------------|
| Frontend       | Streamlit          | Camera UI + Song display |
| Backend        | FastAPI + Uvicorn  | API server               |
| ML Model       | CNN (ONNX format)  | Emotion detection        |
| Hosting        | Render.com         | Backend deployment       |
| Model Training | Google Colab       | FER2013 dataset          |

---

## CNN Model Details

- Dataset: FER2013 (35,000+ facial images)
- Classes: 7 emotions - angry, disgust, fear, happy, neutral, sad, surprise
- Architecture: 3x Conv2D + BatchNorm + MaxPooling + Dense layers
- Format: ONNX (lightweight, no TensorFlow needed at runtime)
- Input: 48x48 grayscale face image
- Output: Emotion probabilities (softmax)

---

## Project Structure

```
Moodetune/
|-- app.py                  # Streamlit frontend
|-- main.py                 # FastAPI backend
|-- motune_model.onnx       # Trained CNN model (ONNX)
|-- motune_model.joblib     # Trained CNN model (joblib)
|-- requirements.txt        # Python dependencies
|-- render.yaml             # Render deployment config
|-- README.md               # This file
```

---

## Tech Stack

- Python - Core language
- TensorFlow/Keras - CNN model training
- ONNX Runtime - Lightweight model inference
- OpenCV - Image preprocessing
- FastAPI - REST API backend
- Uvicorn - ASGI server
- Streamlit - Frontend web app
- Render - Cloud deployment
- Joblib - Model serialization
- Kaggle API - Dataset download

---

## Features

- Real-time facial emotion detection via camera
- 7 emotion classes detection
- Song recommendations based on mood
- Reassess Mood button for re-detection
- Shuffle Songs button
- YouTube links for instant playback
- Mobile friendly
- Android APK available

---

## Run Locally

```bash
git clone https://github.com/parth-eng-123/Moodetune.git
cd Moodetune
pip install -r requirements.txt
streamlit run app.py
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

## Model Training Code

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

## Developer

Parth - GNIOT Capstone Project
GitHub: https://github.com/parth-eng-123

---

## License

This project is for educational purposes as part of the GNIOT Capstone Project.
