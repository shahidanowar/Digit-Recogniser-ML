# 🧠 Handwritten Digit Recognizer

This is a simple web app that uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) drawn by the user on a canvas. It is built using **TensorFlow** for the model and **Streamlit** for the user interface.

---

## 📌 Features

- Train a CNN model on the MNIST dataset  
- Draw digits using an interactive canvas  
- Predict the digit in real time  
- Simple and clean UI powered by Streamlit

---

## 📂 Project Structure

├── train.py # Script to train and save the CNN model
├── app.py # Streamlit app to draw and predict digits
├── digit_model.keras # Saved CNN model
├── requirements.txt # All dependencies
└── README.md # This file

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/digit-recognizer.git
cd digit-recognizer
2. Install dependencies
Create a virtual environment (optional but recommended), then install packages:

bash
Copy
Edit
pip install -r requirements.txt
3. Train the model (optional, already provided)
If you want to retrain the model from scratch:

bash
Copy
Edit
python train.py
This will train a CNN on the MNIST dataset and save the model as digit_model.keras.

4. Run the app
bash
Copy
Edit
streamlit run app.py
🖼️ How to Use
Run the app.

A canvas will appear where you can draw a digit using your mouse or touchpad.

Click the "Predict" button.

The app will show the predicted digit using the trained CNN model.

🛠 Tech Stack
TensorFlow – CNN model for digit classification

Streamlit – Web UI framework

OpenCV – Image processing

NumPy – Data manipulation

Matplotlib – Visualization (used during training)

streamlit-drawable-canvas – Drawing interface

🧠 Model Overview
The model is a simple CNN architecture trained on the MNIST dataset:

Conv2D: 32 filters, 3x3 kernel, ReLU activation

MaxPooling2D: 2x2 pool size

Flatten: to convert 2D to 1D

Dense: 64 units with ReLU activation

Dense: 10 units with softmax for output (digits 0–9)

📷 Example
You draw a digit like "5" on the canvas, and the app predicts it as "5".
(Insert image here if you have a screenshot or GIF of the app.)

📄 License
This project is open-source and free to use under the MIT License.

🙌 Acknowledgements
MNIST dataset by Yann LeCun et al.

Streamlit community for the awesome tools
