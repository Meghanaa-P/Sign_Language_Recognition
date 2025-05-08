# Sign_Language_Recognition



## 📌 Overview

This project is a **real-time sign language recognition system** that uses a webcam to detect hand gestures and convert them into meaningful sentences. It leverages **OpenCV**, **cvzone**, **TensorFlow**, and **Streamlit** to provide an interactive user experience.

## 🔥 Features

- 🖐 **Hand Gesture Detection** using OpenCV & cvzone
- 🤖 **Real-time Sign Recognition** with a trained deep learning model (MobileNetV2)
- 📝 **Sentence Formation** from recognized gestures
- 🖼 **Live Video Stream Processing** via Streamlit
- 📄 **Customizable Dataset** for training new gestures

## 🚀 Technologies Used

- **Python**
- **OpenCV**
- **TensorFlow & Keras**
- **cvzone**
- **Streamlit**
- **NumPy & Pandas**

## 📂 Project Structure

```
├── Project Root
│   ├── dataset.h5                         # Trained Model
│   ├── labels.txt                         # Gesture Labels
│   ├── sentence.csv                       # Sentence Mapping Data
│   ├── Sign_Language_Recognition.py       # Streamlit App
│   ├── requirements.txt                   # Dependencies
│   ├── README.md                          # Project Documentation
│   └── images/                            # Example Images
```

## 🛠 Installation & Setup

### PYTHON 3.10.0 : https://www.python.org/downloads/release/python-3100/

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

## Data Folder Structure
```
Data/
├── Water/
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── ...
├── Rock/
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── ...
└── Paper/
    ├── Image_1.jpg
    ├── Image_2.jpg
    └── ...
```

## 🎯 How to Use

1. Open the Streamlit app.
2. Click **"Start Webcam"** in the sidebar.
3. Show a hand gesture in front of the webcam.
4. Recognized gestures will appear on the screen.
5. Click **"Form Sentence"** to generate a sentence.
6. Click **"Clear"** to reset.

## 📊 Model Training

To train the model with new data:

1. Organize your dataset with labeled gesture images.
2. Modify `train.csv` to map gestures to sentences.
3. Train using MobileNetV2 with `ImageDataGenerator`.
4. Save the model as `dataset.h5`.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📜 License

This project is licensed under the MIT License.

---

🔗 **Connect with Me:**\
📧 Email: [meghanapshetty03@gmail.com](mailto\:meghanapshetty03@gmail.com)\
🔗 LinkedIn: https://www.linkedin.com/in/meghashetty-tech

---

⭐ If you find this project useful, please **star** this repository! ⭐

