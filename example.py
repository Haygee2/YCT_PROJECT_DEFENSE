import os
import sqlite3
import numpy as np
import pytesseract
import streamlit as st
from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
import cv2
import time

# Set up OpenAI API (replace with your API key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Set the path to Tesseract-OCR executable
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
    os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
else:  # Unix/Linux/Mac
    # Use default system path or set accordingly
    pass

# Admin authentication (hardcoded for now)
ADMINS = {"admin": "password"}  # Change credentials as needed

# Database setup
DB_PATH = "students.db"

def init_db():
    """Initialize the database and create the students table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                            matric TEXT PRIMARY KEY,
                            name TEXT,
                            folder TEXT,
                            face_image TEXT)''')
        conn.commit()

def execute_query(query, params=(), fetchone=False, fetchall=False):
    """
    General function to execute queries safely.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)

        if fetchone:
            return cursor.fetchone()
        elif fetchall:
            return cursor.fetchall()
        
        conn.commit()
        return None

def get_student_info(matric):
    """Fetch student details by matric number."""
    return execute_query("SELECT * FROM students WHERE matric = ?", (matric,), fetchone=True)

def store_student(matric, name, folder, face_image_path=""):
    """Stores or updates student information in the database."""
    query = '''INSERT OR REPLACE INTO students (matric, name, folder, face_image) 
               VALUES (?, ?, ?, ?)'''
    execute_query(query, (matric, name, folder, face_image_path))

def get_student_folder(matric, name):
    """Retrieve or create the student’s folder path."""
    student = get_student_info(matric)
    
    if student:
        return student[2]  # Folder path from database
    else:
        folder_path = os.path.join("students_data", f"{name}_{matric[-3:]}")
        os.makedirs(folder_path, exist_ok=True)
        store_student(matric, name, folder_path)  # Store in DB
        return folder_path

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"OCR Error: {e}"

def pdf_to_images(pdf_path):
    """Convert PDF pages to images."""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    
    return images

def save_document(folder, file):
    """Save uploaded documents (PDF or image) and extract text if necessary."""
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.name)

    file_content = file.read()  # Read file content into memory

    with open(file_path, "wb") as f:
        f.write(file_content)

    # Extract text if it's a PDF
    if file.name.endswith(".pdf"):
        images = pdf_to_images(file_path)
        text = "".join(extract_text_from_image(img) for img in images)
        text_file_path = file_path.replace(".pdf", ".txt")

        with open(text_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)

    return file_path

def capture_face_image(matric, name, cam_index=0):
    """Capture a student's face using the webcam and save it."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Create placeholders for the webcam feed and status
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        st.error("Could not open webcam. Try changing the camera index.")
        return False
    
    status_placeholder.info("Looking for face... Please position yourself in frame.")
    
    # Add a stop button
    stop_button = st.button("Stop Capture")
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Failed to capture image.")
            break
            
        # Display the current frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            student_folder = os.path.join("student_folders", f"{name}_{matric}")
            os.makedirs(student_folder, exist_ok=True)
            face_path = os.path.join(student_folder, f"{name}_{matric}_face.jpg")
            
            cv2.imwrite(face_path, face_crop)
            status_placeholder.success(f"Face captured successfully!")
            
            # Show the captured face
            st.image(face_crop, channels="BGR", caption="Captured Face")
            cap.release()
            return True
            
        # Add a small delay to not overwhelm the UI
        time.sleep(0.1)
    
    cap.release()
    return False

def chat_with_ai(query):
    """AI Chatbot interaction using OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.title("YABA COLLEGE OF TECHNOLOGY COMPUTER ENGINEERING DEPARTMENT")

# Login Section
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Admin Login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    
    if st.button("Login"):
        if username in ADMINS and ADMINS[username] == password:
            st.session_state.logged_in = True
            st.success("Login Successful!")
        else:
            st.error("Invalid Credentials")
    st.stop()

# Admin Panel
tab1, tab2 = st.tabs(["Admin Panel", "Student Panel"])

with tab1:
    st.subheader("Admin Panel: Upload Student Documents")
    admin_matric = st.text_input("Enter Student Matric Number:").strip()
    admin_name = st.text_input("Enter Student Name:")
    doc_file = st.file_uploader("Upload Document (Image or PDF)", type=["jpg", "png", "pdf"])

    if st.button("Save Document"):
        if admin_matric and admin_name and doc_file:
            folder = get_student_folder(admin_matric, admin_name)
            save_document(folder, doc_file)
            st.success(f"Data saved for {admin_name} (Matric: {admin_matric}).")
        else:
            st.error("Please enter all required details.")

# Student Panel
with tab2:
    st.subheader("Student Panel: Retrieve Information")
    student_matric = st.text_input("Enter Matric Number to Retrieve Info:").strip()

    if st.button("Get Information"):
        student = get_student_info(student_matric)
        st.write(f"Student Info: {student}")

