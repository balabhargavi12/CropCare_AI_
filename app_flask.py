from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import io
import json
import sqlite3
import hashlib
import time
import base64
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

# Check if PyTorch model file exists but don't import PyTorch yet
PYTORCH_MODEL_EXISTS = os.path.exists("best_optimized_model.pth")
PYTORCH_AVAILABLE = False  # Default to False, will check on demand

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Constants
TF_MODEL_PATH = "best_crop_disease_model.h5"
PYTORCH_MODEL_PATH = "best_optimized_model.pth"
IMG_SIZE = (224, 224)
CHANNELS = 3

# Always use TensorFlow for now until PyTorch issues are resolved
USE_PYTORCH = False
print(f"Using TensorFlow model: {TF_MODEL_PATH}")

# Note about PyTorch model
if PYTORCH_MODEL_EXISTS:
    print(f"PyTorch model found at {PYTORCH_MODEL_PATH}, but using TensorFlow model for now.")
    print("To use PyTorch model, install PyTorch and modify the USE_PYTORCH flag.")

# Class names (same as in app.py)
CLASS_NAMES = [
    "Corn Crop Diseases", "Cotton Crop Diseases", "Fruit Crop Diseases", "Pulse Crop",
    "Rice plant Diseases", "Tobacco Crop Diseases", "Vegetable Crop Diseases", "Wheat Diseases"
]

# Helper functions (copied from app.py)
def ensure_rgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def preprocess_image(pil_img, target_size=IMG_SIZE):
    """Preprocess the image for model input"""
    img = ImageOps.exif_transpose(pil_img)
    img = ensure_rgb(img)

    # letterbox to keep aspect ratio
    img.thumbnail(target_size, Image.LANCZOS)
    # paste centered on white canvas to reach exact target_size
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    x = (target_size[0] - img.size[0]) // 2
    y = (target_size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))
    arr = np.asarray(canvas).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr, canvas  # return processed array and the display image

def softmax(x):
    """Apply softmax function for probability distribution"""
    e = np.exp(x - np.max(x))
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-9, None)

def is_diseased(class_name):
    """Determine if a class indicates disease"""
    tokens = class_name.lower()
    unhealthy_keywords = ["disease", "diseases", "blight", "rust", "mildew", 
                          "leaf spot", "bacterial", "viral", "wilt", "infect"]
    return any(k in tokens for k in unhealthy_keywords)

def crop_type_from_class(class_name):
    """Extract crop type from class name"""
    if "___" in class_name:
        crop = class_name.split("___")[0].strip()
        return crop
    # lightweight heuristics
    words = class_name.replace("plant", "").replace("Crop", "").replace("Diseases", "").replace("Disease", "")
    words = words.replace("_", " ").strip()
    # take first word as crop (works for 'Rice', 'Wheat', 'Corn', etc.)
    return words.split()[0] if words else class_name

# Model loading - cached to avoid reloading
_model_cache = None
def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = TFModel(TF_MODEL_PATH)
    return _model_cache

# TensorFlow model wrapper class
class TFModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def predict(self, image_array, **kwargs):
        return self.model.predict(image_array, **kwargs)

# Database functions
def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, created_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_predictions
                 (username TEXT, image_name TEXT, prediction TEXT, confidence REAL, 
                  is_healthy BOOLEAN, crop_type TEXT, prediction_date TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Create a secure hash of the password"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def auth_user(username, password):
    """Authenticate a user with username and password"""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    stored_password = c.fetchone()
    conn.close()
    return stored_password and stored_password[0] == hash_password(password)

def register_user(username, password):
    """Register a new user"""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users VALUES (?, ?, ?)', 
                 (username, hash_password(password), datetime.now().isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_prediction(username, image_name, prediction, confidence, is_healthy, crop_type):
    """Save prediction results to database"""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('INSERT INTO user_predictions VALUES (?, ?, ?, ?, ?, ?, ?)',
              (username, image_name, prediction, confidence, is_healthy, crop_type, 
               datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_user_predictions(username):
    """Get a user's prediction history"""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''SELECT image_name, prediction, confidence, is_healthy, crop_type, prediction_date 
                 FROM user_predictions WHERE username=? ORDER BY prediction_date DESC''', (username,))
    predictions = c.fetchall()
    conn.close()
    return predictions

# Initialize database
init_db()

# Routes
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if auth_user(data['username'], data['password']):
        session['username'] = data['username']
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if register_user(data['username'], data['password']):
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Username already exists'})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('home'))
    predictions = get_user_predictions(session['username'])
    return render_template('history.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'})
    
    try:
        # Process image
        image = Image.open(file.stream)
        arr, _ = preprocess_image(image)
        
        # Make prediction
        model = get_model()
        prediction = model.predict(arr, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = CLASS_NAMES[class_idx]
        is_healthy_flag = not is_diseased(predicted_class)
        crop_type = crop_type_from_class(predicted_class)
        
        # Save prediction
        save_prediction(
            session['username'],
            file.filename,
            predicted_class,
            confidence,
            is_healthy_flag,
            crop_type
        )
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': f"{confidence:.2%}",
            'crop_type': crop_type,
            'is_healthy': is_healthy_flag
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # Create static and templates folders if they don't exist
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Starting CropCare AI Flask App...")
    print("Visit http://127.0.0.1:5000 to access the application")
    app.run(debug=True)
