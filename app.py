import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
import re
from PIL import Image
import tempfile

# Set page config first
st.set_page_config(page_title="Face Recognition App", layout="wide")

# Function to clean filename
def clean_name(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'\s*\(\d+\)\s*', '', name)
    return name.strip()

# Load known faces with progress indicator
@st.cache_resource
def load_known_faces(known_faces_folder="known_faces"):
    known_face_encodings = []
    known_face_names = []
    
    if os.path.exists(known_faces_folder):
        files = [f for f in os.listdir(known_faces_folder) 
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if files:
            progress_bar = st.progress(0)
            total_files = len(files)
            
            for i, filename in enumerate(files):
                img_path = os.path.join(known_faces_folder, filename)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(clean_name(filename))
                
                progress_bar.progress((i + 1) / total_files)
            
            st.success(f"Loaded {len(known_face_encodings)} known faces")
        else:
            st.warning("No image files found in the known_faces folder")
    else:
        st.error(f"Folder '{known_faces_folder}' not found")
    
    return known_face_encodings, known_face_names

# Process uploaded image
def process_uploaded_image(uploaded_file, known_face_encodings, known_face_names):
    try:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            color = (255, 0, 0)  # Red
            
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < 0.5:
                        name = known_face_names[best_match_index]
                        color = (0, 255, 0)  # Green
            
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return Image.fromarray(image)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Live face detection
def live_face_detection(known_face_encodings, known_face_names):
    st.write("Starting live detection...")
    
    run_detection = st.checkbox("Keep detection running", value=True)
    video_placeholder = st.empty()
    stop_button = st.button("Stop Detection")
    
    video_capture = cv2.VideoCapture(0)
    
    while run_detection and not stop_button:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            color = (255, 0, 0)
            
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < 0.5:
                        name = known_face_names[best_match_index]
                        color = (0, 255, 0)
            
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
    
    video_capture.release()
    st.write("Detection stopped")

# Main app
def main():
    st.title("Face Recognition App")
    st.markdown("""
    This app recognizes faces from your `known_faces` folder and can detect faces in:
    - Uploaded images
    - Live webcam feed
    """)
    
    # Initialize session state
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Upload Image", "Live Detection"])
    
    with tab1:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.write("Processing image...")
            result = process_uploaded_image(uploaded_file, known_face_encodings, known_face_names)
            if result:
                st.image(result, caption="Detected Faces", use_column_width=True)
    
    with tab2:
        st.header("Live Face Detection")
        st.write("Click below to start live face detection from your webcam")
        
        if st.button("Start Live Detection"):
            live_face_detection(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()