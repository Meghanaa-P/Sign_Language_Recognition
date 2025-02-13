import os
import streamlit as st
import cv2
import numpy as np
import math
import pandas as pd
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Streamlit UI setup
st.set_page_config(page_title="Sign Language Recognition", layout="wide")
st.title("🖐️ Real-time Sign Language to Sentence Conversion")

# Sidebar controls
st.sidebar.header("Control Panel")
run = st.sidebar.checkbox("Start Webcam", value=False)
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. Position your hand in front of the webcam.")
st.sidebar.markdown("2. Ensure your hand is clearly visible.")
st.sidebar.markdown("3. Use the 'Form Sentence' and 'Clear' buttons as needed.")

# Placeholder for webcam output
FRAME_WINDOW = st.image([])

# Initialize session state parameters
for key in ["recognized_words", "frame_count", "previous_word", "sentence"]:
    st.session_state.setdefault(key, []) if key == "recognized_words" else st.session_state.setdefault(key, "")

# Load the classifier and labels
try:
    detector = HandDetector(maxHands=1)
    classifier = Classifier(
        r"C:\Users\MEGHANA\OneDrive\Desktop\Project\dataset.h5",
        r"C:\Users\MEGHANA\OneDrive\Desktop\Project\after\Model2\labels.txt"
    )
except Exception as e:
    st.error(f"Error loading the model or labels: {e}")
    st.stop()

# Load sentence data from CSV file
def load_sentence_data():
    sentence_data = {}
    try:
        df = pd.read_csv(r'C:\Users\MEGHANA\OneDrive\Desktop\Project\after\train.csv', on_bad_lines='skip')
        for _, row in df.iterrows():
            input_words = str(row.get('input', '')).lower()  # Treat input as a single string for the key
            output_sentence = str(row.get('output', ''))
            sentence_data[input_words] = output_sentence  # Store sentence by input key
    except FileNotFoundError:
        st.error("CSV file not found.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please check its format.")
    return sentence_data

sentences_data = load_sentence_data()

# Hand gesture labels
labels = ["Angry", "Bad", "Family", "Food", "Friend", "Go", "Good", "Goodbye", "Happy", "Hello", "Help", "How", "Love", "Name", "Nice to meet you", "No", "Please", "Sad", "Sorry", "Stop", "Thank you", "Want", "Washroom", "Water", "What", "When", "Where", "Who", "Why", "Yes"]

# Function to get the most likely sentence using the recognized words
def get_most_likely_sentence(words, sentence_data):
    # Combine the recognized words into a single string as a key
    key = ' '.join(words).lower()  # Use a space to join the words and convert to lowercase
    matched_sentence = sentence_data.get(key, None)  # Search for the key in sentence_data
    
    # If a sentence is found for the combined words, return it
    if matched_sentence:
        return matched_sentence
    else:
        # Fallback if no matching sentence is found
        return "Could not form a sentence from the recognized words."

# Main loop for capturing and processing frames
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
    else:
        sentence_placeholder, final_sentence_placeholder, recognized_words_placeholder = st.empty(), st.empty(), st.empty()
        form_sentence_button, clear_sentence_button = st.button("Form Sentence"), st.button("Clear Sentence")

        while run:
            success, img = cap.read()
            if not success:
                st.error("Unable to capture the frame.")
                break

            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((300, 300, 3), np.uint8) * 255
                y1, y2, x1, x2 = max(0, y - 20), min(img.shape[0], y + h + 20), max(0, x - 20), min(img.shape[1], x + w + 20)
                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size != 0:
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        wCal = math.ceil((300 / h) * w)
                        imgResize = cv2.resize(imgCrop, (wCal, 300))
                        imgWhite[:, (300 - wCal) // 2:(300 - wCal) // 2 + wCal] = imgResize
                    else:
                        hCal = math.ceil((300 / w) * h)
                        imgResize = cv2.resize(imgCrop, (300, hCal))
                        imgWhite[(300 - hCal) // 2:(300 - hCal) // 2 + hCal, :] = imgResize
                    
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    predicted_word = labels[index]
                    if st.session_state.frame_count % 20 == 0 and predicted_word != st.session_state.previous_word:
                        st.session_state.recognized_words.append(predicted_word)
                        st.session_state.previous_word = predicted_word
                    
                    cv2.putText(imgOutput, predicted_word, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 6)
                    cv2.putText(imgOutput, predicted_word, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 255), 4)

            FRAME_WINDOW.image(imgOutput, channels="BGR")
            recognized_words_placeholder.write("### Current Words: " + " ".join(st.session_state.recognized_words))
            st.session_state.frame_count += 1

            if form_sentence_button:
                st.session_state.sentence = get_most_likely_sentence(st.session_state.recognized_words, sentences_data)
                final_sentence_placeholder.write("### Most Likely Sentence: " + st.session_state.sentence)

            if clear_sentence_button:
                st.session_state.update({"recognized_words": [], "sentence": "", "previous_word": None})
                recognized_words_placeholder.write("### Current Words: ")
                final_sentence_placeholder.write("### Most Likely Sentence: ")

        cap.release()
        cv2.destroyAllWindows()
else:
    st.session_state.update({"recognized_words": [], "frame_count": 0, "previous_word": None, "sentence": ""})
