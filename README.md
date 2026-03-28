# Disaster Detection System using Deep Learning

This project is a deep learning-based system that detects disasters such as Flood, Drought, and Wildfire from images using a trained CNN model.

The system supports both full-image classification and patch-wise analysis for large images.

---

##  Features

- Image-based disaster classification
- Supports multiple disaster types:
  - Drought
  - Flood
  - Wildfire
  - Normal
- Patch-wise analysis for large images
- Heatmap visualization
- Confidence scoring
- Confusion matrix and classification report
- Downloadable results

---

##  Model

- Trained CNN model (`.h5`)
- Input size: 128x128
- Multi-class classification

---

##  Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn

---

##  Project Structure
dataset/ # Training dataset
model/ # Trained model (.h5)
app.py # Streamlit UI
train_model.R # Training script
patch_detection.ipynb # Experiment notebook
requirements.txt # Dependencies


---

##  How to Run


pip install -r requirements.txt
streamlit run app.py

Use Cases
Disaster monitoring systems
Environmental analysis
Satellite image analysis
Early warning systems

Future Improvements
Real-time disaster detection
Integration with satellite APIs
Model optimization
Deployment on cloud

Author

Linsha Bangera
Computer Science & Data Science Engineering

